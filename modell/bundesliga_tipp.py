#!/usr/bin/env python3
"""
Bundesliga-Tipp – statistische Vorhersage von Bundesliga-Spielen.

Modell: Dixon-Coles-Poisson-Modell mit zeitlich gewichteten Daten.
  * Jedes Team hat eine Angriffs- und eine Abwehrstärke.
  * Erwartete Tore Heim = exp(Liga + Heimvorteil + Angriff_Heim + Abwehr_Gast)
    Erwartete Tore Gast = exp(Liga + Angriff_Gast + Abwehr_Heim)
  * Die Dixon-Coles-Korrektur (rho) passt die Wahrscheinlichkeit knapper
    Ergebnisse (0:0, 1:0, 0:1, 1:1) an.
  * Neuere Spiele zählen mehr als alte (exponentieller Zerfall, Parameter xi).

Lernen aus neuen Ergebnissen:
  * Bei jedem Start werden die neuesten Ergebnisse von OpenLigaDB geladen und
    das Modell wird mit ihnen neu trainiert.
  * Sobald ein neuer Spieltag fertig ist, werden die Hyperparameter
    (Zeitgewichtung xi, Regularisierung) per Walk-Forward-Backtest neu
    optimiert: das Modell sagt vergangene Spieltage "blind" voraus und wählt
    die Einstellung, die am besten getroffen hätte.
  * Alle Vorhersagen werden gespeichert und nach dem Spiel ausgewertet,
    so siehst du, wie gut das Modell wirklich ist.

Daten: OpenLigaDB (https://www.openligadb.de), kostenlos, ohne API-Key.
Es werden 1. und 2. Bundesliga verwendet, damit auch Aufsteiger eine
sinnvolle Stärke-Schätzung bekommen.
"""

from __future__ import annotations

import argparse
import bisect
import difflib
import html
import json
import math
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

API = "https://api.openligadb.de"
LEAGUES = ("bl1", "bl2")
SEASONS_BACK = 5            # aktuelle + 5 vergangene Saisons (ab 2021/22): im Test deutlich genauere Prozente
MAX_GOALS = 10
DATA_DIR = Path(__file__).resolve().parent / "daten"
DEFAULT_PARAMS = {"xi": 0.0020, "reg": 4.0, "wxg": 0.65}
XI_GRID = [0.0007, 0.0010, 0.0016]
REG_GRID = [1.5, 3.0, 5.0]
WXG_GRID = [0.5]            # Anteil xG am Lernziel: im Test (3 Saisons) 0,5 mit den meisten exakten Treffern
BACKTEST_MATCHDAYS = 30   # längerer Zeitraum = stabilere Auswahl (getestet: 12 war zu verrauscht)
WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


def fmt_time(dt: datetime, year: bool = False) -> str:
    loc = dt.astimezone()
    return f"{WEEKDAYS[loc.weekday()]} {loc.strftime('%d.%m.%Y' if year else '%d.%m.')} {loc.strftime('%H:%M')}"


# --------------------------------------------------------------------------
# Daten
# --------------------------------------------------------------------------

@dataclass
class Match:
    id: int
    league: str
    season: int
    matchday: int
    kickoff: datetime          # UTC
    home_id: int
    away_id: int
    home: str
    away: str
    finished: bool
    hg: int | None = None
    ag: int | None = None
    home_full: str = ""
    away_full: str = ""


def current_season(now: datetime) -> int:
    return now.year if now.month >= 7 else now.year - 1


def _get_json(url: str, retries: int = 3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                       "User-Agent": "bundesliga-tipp/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Konnte {url} nicht laden: {last}")


def _final_score(raw: dict):
    res = raw.get("matchResults") or []
    if not res:
        return None
    end = [r for r in res if r.get("resultTypeID") == 2]
    r = end[0] if end else max(res, key=lambda r: r.get("resultOrderID", 0))
    return int(r["pointsTeam1"]), int(r["pointsTeam2"])


def _parse(raw: dict, league: str, season: int) -> Match:
    ko = raw.get("matchDateTimeUTC") or raw["matchDateTime"]
    kickoff = datetime.fromisoformat(ko.replace("Z", "+00:00"))
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    score = _final_score(raw) if raw.get("matchIsFinished") else None
    t1, t2 = raw["team1"], raw["team2"]
    return Match(
        id=raw["matchID"], league=league, season=season,
        matchday=raw["group"]["groupOrderID"], kickoff=kickoff,
        home_id=t1["teamId"], away_id=t2["teamId"],
        home=t1.get("shortName") or t1["teamName"],
        away=t2.get("shortName") or t2["teamName"],
        finished=score is not None,
        hg=score[0] if score else None, ag=score[1] if score else None,
        home_full=t1["teamName"], away_full=t2["teamName"],
    )


def load_matches(now: datetime, offline: bool = False) -> list[Match]:
    """Lädt alle Spiele. Abgeschlossene Saisons werden dauerhaft gecacht,
    die laufende Saison wird bei jedem Start aktualisiert."""
    cache = DATA_DIR / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cur = current_season(now)
    matches: list[Match] = []
    for league in LEAGUES:
        for season in range(cur - SEASONS_BACK, cur + 1):
            f = cache / f"{league}_{season}.json"
            raw = None
            if f.exists() and (season < cur or offline):
                raw = json.loads(f.read_text("utf-8"))
            if raw is None:
                try:
                    raw = _get_json(f"{API}/getmatchdata/{league}/{season}")
                    f.write_text(json.dumps(raw), "utf-8")
                except RuntimeError as e:
                    if f.exists():
                        print(f"Warnung: {e} – nutze gespeicherte Daten.", file=sys.stderr)
                        raw = json.loads(f.read_text("utf-8"))
                    else:
                        raise
            matches += [_parse(m, league, season) for m in raw]
    matches.sort(key=lambda m: m.kickoff)
    return matches


# --------------------------------------------------------------------------
# Spielstatistiken (ESPN): Schüsse, Ballbesitz, Pässe, Defensivaktionen
# --------------------------------------------------------------------------

ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/ger.1"
STAT_SEASONS_BACK = 2       # Statistiken für aktuelle + zwei vorherige Saisons (1. Liga)
STAT_KEYS = {"possessionPct": "ballbesitz", "totalShots": "schuesse", "shotsOnTarget": "aufs_tor",
             "penaltyKickShots": "elfmeter", "totalPasses": "paesse", "totalTackles": "tacklings",
             "interceptions": "ballgewinne", "foulsCommitted": "fouls"}
STAT_FIELDS = list(STAT_KEYS.values())


def _norm(s: str) -> str:
    s = s.lower()
    for a, b in (("ü", "u"), ("ö", "o"), ("ä", "a"), ("ß", "ss"), ("é", "e")):
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _sim(m_names, name):
    return max(difflib.SequenceMatcher(None, _norm(n), _norm(name)).ratio() for n in m_names if n)


SHOT_TYPES = ("Shot Off Target", "Shot Blocked", "Shot On Target", "Shot Hit Woodwork", "Goal",
              "Goal - Header", "Goal - Volley", "Goal - Free-kick", "Penalty - Scored", "Penalty - Saved",
              "Penalty - Missed")
EVENT_CODES = {"Foul": "F", "Corner Awarded": "C", "Offside": "O", "Handball": "F", "Pass": "P"}


EV_VERSION = 3   # 3: Merkmale aus dem Kommentar, alte ESPN-Koordinaten ungerundet umgerechnet, Datenquelle markiert
SHOT_ZONES = (("six yard box", "F"), ("very close range", "N"), ("centre of the box", "M"), ("side of the box", "D"),
              ("difficult angle", "W"), ("outside the box", "A"), ("more than 35 yards", "L"),
              ("more than 40 yards", "L"), ("long range", "L"))


def shot_flags(typ: str, text: str) -> str:
    """Merkmale eines Schusses aus dem ESPN-Kommentar – nur, was vor dem Schuss feststeht
    (Ort, Körperteil, Art der Vorlage), nie der Ausgang."""
    t = text.lower()
    pre = t.split(" is ")[0] if " is " in t else t
    vor = t.split("assisted")[0]
    fl = ""
    if "Penalty" in typ or "penalty" in t:
        fl += "E"
    if typ == "Goal - Header" or "header" in pre:
        fl += "H"
    if typ == "Goal - Free-kick" or "free kick" in vor:
        fl += "K"
    for key, code in (("with a cross", "X"), ("through ball", "T"), ("fast break", "B"), ("following a corner", "C"),
                      ("set piece", "P"), ("headed pass", "Q")):
        if key in t:
            fl += code
    if "assisted by" not in t:
        fl += "U"
    for key, code in SHOT_ZONES:
        if (key in pre or key in vor) and code not in fl:
            fl += code
    return fl


def _parse_events(summary: dict, names: dict) -> list:
    """Spielereignisse mit Feldposition. Kompakt: [team, typ, x, y, sekunde, merkmale, tor].
    team = 'h'/'a' (wer die Aktion ausführt; bei Fouls das foulende Team), x/y in Prozent aus
    Sicht dieses Teams (x = 100 ist das gegnerische Tor)."""
    out, seen = [], set()
    for e in summary.get("commentary", []):
        p = e.get("play")
        if not p or p.get("id") in seen:
            continue
        seen.add(p.get("id"))
        t = p.get("type", {}).get("text", "")
        team = names.get(p.get("team", {}).get("displayName"))
        if team is None:
            continue
        sec = float(p.get("clock", {}).get("value") or 0)
        if t in SHOT_TYPES:
            fl = shot_flags(t, p.get("text", ""))
            goal = 1 if t.startswith("Goal") or t == "Penalty - Scored" else 0
            x, y = p.get("fieldPositionX"), p.get("fieldPositionY")
            if "E" in fl:
                x, y = 88.6, 50.0
            if x is None or y is None:
                continue
            out.append([team, "S", float(x), float(y), sec, fl, goal])   # gerundet wird erst nach der Umrechnung
        elif t in EVENT_CODES and p.get("fieldPositionX") is not None:
            out.append([team, EVENT_CODES[t], float(p["fieldPositionX"]),
                        float(p.get("fieldPositionY", 50)), sec, "", 0])
    return normalize_events(out)


def normalize_events(ev: list) -> list:
    """Bringt Positionen auf eine Konvention: Prozent, x = 100 ist das Tor, auf das geschossen wird.
    Bis 2025/26 lieferte ESPN Anteile (0–1), bei Schüssen ist x dort der Abstand zum Tor als Anteil der
    halben Feldlänge (am Kommentartext geprüft: „centre of the box“ ≈ 0,2 → gut 10 m). Für die übrigen
    Ereignisse dieser Spiele ist die Konvention nicht geprüft; sie werden weggelassen.
    Elfmeter stehen fest auf dem Punkt."""
    free = [e for e in ev if not (e[1] == "S" and "E" in e[5])]
    alt = bool(free) and max(max(e[2], e[3]) for e in free) <= 1.0
    out = []
    for e in ev:
        e = list(e)
        if e[1] == "S" and "E" in e[5]:
            e[2], e[3] = 88.6, 50.0
        elif alt:
            if e[1] != "S" or (e[2] == 0 and e[3] == 0):
                continue
            e[2], e[3] = round(min(max(100 - 50 * e[2], 0), 100), 1), round(min(max(100 * e[3], 0), 100), 1)
        else:
            e[2], e[3] = round(min(max(e[2], 0), 100), 1), round(min(max(e[3], 0), 100), 1)
            if e[1] == "S" and "Z" not in e[5]:
                e[5] += "Z"     # neue ESPN-Datenquelle (ab 2026/27): mehr Schüsse je Spiel, seltener ein Tor je Schuss
        out.append(e)
    return out


def _parse_lineups(summary: dict) -> dict:
    """Spieltagskader je Team: [spieler_id, name, startelf, eingewechselt, tore, vorlagen, gelb, rot]."""
    out = {}
    for r in summary.get("rosters", []) or []:
        side = "h" if r.get("homeAway") == "home" else "a"
        rows = []
        for p in r.get("roster", []) or []:
            ath = p.get("athlete", {})
            st = {x.get("name"): x.get("value") or 0 for x in p.get("stats", []) or []}
            rows.append([str(ath.get("id")), ath.get("shortName") or ath.get("displayName") or "",
                         1 if p.get("starter") else 0, 1 if p.get("subbedIn") else 0,
                         int(st.get("totalGoals", 0)), int(st.get("goalAssists", 0)),
                         int(st.get("yellowCards", 0)), int(st.get("redCards", 0))])
        if rows:
            out[side] = rows
    return out if len(out) == 2 else {}


def _parse_summary(summary: dict):
    comp = summary["header"]["competitions"][0]
    side = {c["team"]["id"]: c["homeAway"] for c in comp["competitors"]}
    names = {c["team"]["displayName"]: ("h" if c["homeAway"] == "home" else "a") for c in comp["competitors"]}
    out = {}
    for t in summary.get("boxscore", {}).get("teams", []):
        ha = t.get("homeAway") or side.get(t["team"]["id"])
        vals = {}
        for s in t.get("statistics", []):
            k = STAT_KEYS.get(s.get("name"))
            if k:
                try:
                    vals[k] = float(s["displayValue"])
                except (TypeError, ValueError):
                    pass
        if ha in ("home", "away") and all(k in vals for k in STAT_FIELDS):
            out["h" if ha == "home" else "a"] = vals
    if len(out) != 2:
        return None
    out["ev"] = _parse_events(summary, names)
    out["evv"] = EV_VERSION
    out["lu"] = _parse_lineups(summary)
    return out


def load_stats(matches: list[Match], now: datetime, offline: bool = False,
               imported: dict | None = None) -> dict:
    """Lädt Spielstatistiken der 1. Bundesliga von ESPN und ordnet sie den
    OpenLigaDB-Spielen zu (gleicher Tag, ähnliche Teamnamen, gleiches Ergebnis)."""
    f = DATA_DIR / "cache" / "espn_stats.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    stats = json.loads(f.read_text("utf-8")) if f.exists() else {}
    if imported:
        stats.update(imported)
    cur = current_season(now)
    need = [m for m in matches if m.league == "bl1" and m.finished
            and m.season >= cur - STAT_SEASONS_BACK
            and (str(m.id) not in stats or "ev" not in stats[str(m.id)] or "lu" not in stats[str(m.id)]
                 or stats[str(m.id)].get("evv") != EV_VERSION)]
    if need and not offline:
        by_date: dict[str, list[Match]] = {}
        for m in need:
            by_date.setdefault(m.kickoff.strftime("%Y%m%d"), []).append(m)

        def fetch_day(date):
            try:
                return date, _get_json(f"{ESPN}/scoreboard?dates={date}").get("events", [])
            except RuntimeError:
                return date, []

        with ThreadPoolExecutor(6) as ex:
            days = dict(ex.map(fetch_day, by_date))
        pairs = []
        for date, ms in by_date.items():
            cand = []
            for e in days.get(date, []):
                comp = e["competitions"][0]
                if not comp["status"]["type"].get("completed"):
                    continue
                cs = {c["homeAway"]: c for c in comp["competitors"]}
                if "home" not in cs or "away" not in cs:
                    continue
                cand.append((e["id"], cs["home"]["team"]["displayName"], cs["away"]["team"]["displayName"],
                             str(cs["home"].get("score")), str(cs["away"].get("score"))))
            for m in ms:
                scored = [(_sim((m.home, m.home_full), c[1]) + _sim((m.away, m.away_full), c[2]), c)
                          for c in cand if c[3] == str(m.hg) and c[4] == str(m.ag)]
                if scored:
                    s, c = max(scored, key=lambda t: t[0])
                    if s >= 0.9:
                        pairs.append((m, c[0]))

        def fetch_sum(pair):
            m, eid = pair
            try:
                return m, _parse_summary(_get_json(f"{ESPN}/summary?event={eid}"))
            except (RuntimeError, KeyError, IndexError):
                return m, None

        with ThreadPoolExecutor(6) as ex:
            for m, st in ex.map(fetch_sum, pairs):
                if st:
                    stats[str(m.id)] = st
        f.write_text(json.dumps(stats), "utf-8")
        missing = len(need) - sum(str(m.id) in stats for m in need)
        if missing:
            print(f"Hinweis: für {missing} Spiele keine ESPN-Statistik gefunden.", file=sys.stderr)
    return stats


# --------------------------------------------------------------------------
# Merkmale: xG (aus Schüssen geschätzt), Form-Index, Tabellenplatz
# --------------------------------------------------------------------------

FORM_GAMES = 4      # Formfenster; im Test an 358 Spielen besser als 6 oder 10
AUSFALL_SPIELE = 10  # so viele letzte Spiele bestimmen, wer Schlüsselspieler ist
# Gewichte aus dem Walk-forward-Test (gewichte_test.py / gewichte_test2.py): die aus den Daten
# gelernten Einflüsse zeigten xG und Tabellenplatz als stärkste Faktoren, Schüsse und PPDA ohne Nutzen.
FORM_WEIGHTS = {"xg": 0.55, "schuesse": 0.0, "xt": 0.10, "feldkontrolle": 0.10,
                "ballbesitz": 0.05, "ppda": 0.0, "tabelle": 0.20}
FORM_LABELS = {"xg": "xG-Differenz", "schuesse": "Schussdifferenz", "xt": "Gefahr (xT)",
               "feldkontrolle": "Feldkontrolle", "ballbesitz": "Ballbesitz",
               "ppda": "Pressing (PPDA)", "tabelle": "Tabellenplatz"}
PENALTY_XG = 0.76


# --------------------------------------------------------------------------
# Raumdaten: Positions-xG, Expected Threat (Annäherung), Feldkontrolle (Annäherung)
# --------------------------------------------------------------------------

PITCH_L, PITCH_W = 105.0, 68.0
XG_SCHUSS_ANTEIL = 0.5  # xG je Spiel: halb Schuss-Modell, halb Schätzung aus Schusszahlen (inkl. aufs Tor); im Test so gut wie bisher, rein aus Schüssen etwas schlechter
XG_RIDGE = 10.0    # im Test (6 Saisons, je eine herausgehalten) zwischen 1 und 50 praktisch gleich gut
XT_XBINS = [0, 33.3, 50, 66.7, 83.3, 100.01]     # Zonen in Spielrichtung (Prozent der Feldlänge)
XT_YBINS = [0, 25, 75, 100.01]                   # Flügel / Mitte / Flügel
XT_WINDOW = 120                                   # Sekunden: Tor innerhalb dieser Zeit = Gefahr


def shot_geometry(x: float, y: float):
    dx = max((100 - x) / 100 * PITCH_L, 0.3)
    dy = (y - 50) / 100 * PITCH_W
    dist = math.hypot(dx, dy)
    half = 7.32 / 2
    ang = math.atan2(7.32 * dx, dx * dx + dy * dy - half * half)
    if ang < 0:
        ang += math.pi
    return dist, ang


def shot_row(e) -> list[float]:
    """Merkmale eines Schusses (ohne Konstante): Distanz und Winkel nichtlinear, Körperteil,
    Vorlagenart und Ortsangabe aus dem Kommentar. Getestet in xg_test.py."""
    d, a = shot_geometry(e[2], e[3])
    fl = e[5]
    h = float("H" in fl)
    return [d, math.log(d), d * d / 100, a, a * a, a * d / 10, h, h * d / 10, float("K" in fl),
            float("X" in fl), float("T" in fl), float("B" in fl), float("C" in fl), float("P" in fl),
            float("Q" in fl), float("U" in fl), float("F" in fl), float("N" in fl), float("M" in fl),
            float("D" in fl), float("W" in fl), float("A" in fl), float("L" in fl), float("Z" in fl)]


def fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1.0, iters: int = 100) -> np.ndarray:
    """Ridge-logistische Regression (erste Spalte = Konstante, ohne Strafe); Newton mit
    Schrittweitenkontrolle, damit sie auch bei vielen, verwandten Merkmalen sicher konvergiert."""
    w = np.zeros(X.shape[1])
    w[0] = math.log(max(y.mean(), 1e-3) / max(1 - y.mean(), 1e-3))
    R = np.eye(X.shape[1]) * ridge
    R[0, 0] = 0

    def loss(v):
        z = X @ v
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * v @ R @ v)
    f = loss(w)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + R @ w
        Hm = X.T @ (X * (p * (1 - p))[:, None]) + R + 1e-6 * np.eye(len(w))
        step = np.linalg.solve(Hm, g)
        t, fn = 1.0, f
        while t > 1e-4:
            fn = loss(w - t * step)
            if fn <= f:
                break
            t /= 2
        w, f_old, f = w - t * step, f, fn
        if f_old - f < 1e-9:
            break
    return w


def events_ok(ev: list) -> bool:
    """Nur Spiele mit genauen, einheitlich ausgerichteten Positionen verwenden
    (ältere ESPN-Daten haben grobe 10er-Raster und wechselnde Richtung)."""
    shots = [e for e in ev if e[1] == "S" and "E" not in e[5]]
    if len(shots) < 4 or sum(e[1] != "S" for e in ev) < 20:
        return False
    forward = sum(e[2] >= 50 for e in shots) / len(shots)
    coarse = sum(e[2] % 10 == 0 and e[3] % 10 == 0 for e in ev) / max(len(ev), 1)
    return forward >= 0.8 and coarse < 0.5


class Spatial:
    """Lernt aus allen Ereignissen mit Position ein xG-Modell und Zonenwerte (xT-Annäherung)."""

    def __init__(self, events_by_match: dict[str, list], shots_by_match: dict[str, list] | None = None):
        src = events_by_match if shots_by_match is None else shots_by_match
        shots = [e for ev in src.values() for e in ev if e[1] == "S"]
        np_shots = [e for e in shots if "E" not in e[5]]
        pens = [e for e in shots if "E" in e[5]]
        self.n_shots = len(shots)
        self.penalty_xg = (sum(e[6] for e in pens) + 0.76 * 20) / (len(pens) + 20)
        if len(np_shots) >= 500:
            X = np.array([shot_row(e) for e in np_shots])
            y = np.array([e[6] for e in np_shots], float)
            mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
            Xs = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
            self.w, self.mu, self.sd = fit_logistic(Xs, y, ridge=XG_RIDGE), mu, sd
            p = self._p(X)
            self.calib = (float(p.sum()), float(y.sum()))
        else:
            self.w = None
            self.calib = (0.0, 0.0)

        # Zonenwerte: Wie oft trifft das Team mit Ballbesitz in dieser Zone innerhalb von 2 Minuten?
        nx, ny = len(XT_XBINS) - 1, len(XT_YBINS) - 1
        hits, cnt = np.zeros((nx, ny)), np.zeros((nx, ny))
        for ev in events_by_match.values():
            goals = {"h": [e[4] for e in ev if e[1] == "S" and e[6] and e[0] == "h"],
                     "a": [e[4] for e in ev if e[1] == "S" and e[6] and e[0] == "a"]}
            for e in ev:
                pos = self.possession(e)
                if pos is None:
                    continue
                team, x, yy = pos
                i, j = self.zone(x, yy)
                cnt[i, j] += 1
                hits[i, j] += any(e[4] < g <= e[4] + XT_WINDOW for g in goals[team])
        base = hits.sum() / max(cnt.sum(), 1)
        self.xt_grid = (hits + 25 * base) / (cnt + 25)
        self.xt_counts = cnt

    def _p(self, X: np.ndarray) -> np.ndarray:
        Xs = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        return 1 / (1 + np.exp(-np.clip(Xs @ self.w, -30, 30)))

    def shot_xg(self, e) -> float:
        if "E" in e[5]:
            return self.penalty_xg
        if self.w is None:
            return 0.1
        return float(self._p(np.array([shot_row(e)]))[0])

    @staticmethod
    def zone(x: float, y: float):
        i = min(max(bisect.bisect_right(XT_XBINS, x) - 1, 0), len(XT_XBINS) - 2)
        j = min(max(bisect.bisect_right(XT_YBINS, y) - 1, 0), len(XT_YBINS) - 2)
        return i, j

    @staticmethod
    def possession(e):
        """Wer hat bei dieser Aktion den Ball, und wo (aus dessen Sicht)? Schüsse zählen nicht (das ist xG)."""
        code = e[1]
        if code == "F":       # Foul: das gefoulte Team bekommt den Ball
            return ("a" if e[0] == "h" else "h"), 100 - e[2], 100 - e[3]
        if code in ("C", "O", "P"):
            return e[0], e[2], e[3]
        return None

    def match_values(self, ev: list) -> dict:
        xg = {"h": 0.0, "a": 0.0}
        xt = {"h": 0.0, "a": 0.0}
        xs = []
        for e in ev:
            if e[1] == "S":
                xg[e[0]] += self.shot_xg(e)
            pos = self.possession(e)
            if pos is not None:
                team, x, y = pos
                i, j = self.zone(x, y)
                xt[team] += float(self.xt_grid[i, j])
            xs.append(e[2] if e[0] == "h" else 100 - e[2])   # Ballposition aus Sicht der Heimelf
        terr = float(np.mean(xs)) if xs else 50.0
        return {"xg": xg, "xt": xt, "territorium_heim": terr}


# Von Hand eingetragene FotMob-xG: zählen anfangs wenig, mit mehr Einträgen mehr
FOTMOB_MIN = 0.20      # Gewicht bei den ersten Einträgen
FOTMOB_MAX = 0.70      # Höchstgewicht
FOTMOB_FULL = 150      # ab so vielen eingetragenen Spielen gilt das Höchstgewicht


def fotmob_weight(n: int) -> float:
    if n <= 0:
        return 0.0
    return FOTMOB_MIN + (FOTMOB_MAX - FOTMOB_MIN) * min(1.0, n / FOTMOB_FULL)


def _num(v):
    if v is None or v == "":
        return None
    try:
        x = float(str(v).replace(",", "."))
    except ValueError:
        return None
    return x if 0 <= x <= 15 else None


def load_fotmob(path: Path) -> dict[int, tuple[float, float]]:
    """Liest eingetragene FotMob-xG. Akzeptiert eine JSON-Liste, {rows: [...]} oder einen
    Ordner mit JSON-Dateien je Dokument (auch in der Form {id, data: {...}})."""
    objs = []
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    for f in files:
        d = json.loads(f.read_text("utf-8"))
        if isinstance(d, dict) and "rows" in d:
            d = d["rows"]
        if isinstance(d, dict) and "documents" in d:
            d = d["documents"]
        objs += d if isinstance(d, list) else [d]
    out = {}
    for o in objs:
        if not isinstance(o, dict):
            continue
        data = o.get("data") if isinstance(o.get("data"), dict) else o
        mid = data.get("match_id", o.get("id", o.get("doc_id")))
        try:
            mid = int(mid)
        except (TypeError, ValueError):
            continue
        h, a = _num(data.get("xg_heim")), _num(data.get("xg_gast"))
        if h is not None and a is not None:
            out[mid] = (h, a)
    return out


class Features:
    def __init__(self, matches: list[Match], stats: dict, fotmob: dict | None = None):
        bl1 = sorted([m for m in matches if m.league == "bl1" and m.finished], key=lambda m: m.kickoff)
        sm = [(m, stats[str(m.id)]) for m in bl1 if str(m.id) in stats]
        self.n_stat_matches = len(sm)
        known = {m.id for m in bl1}
        self.fotmob = {k: v for k, v in (fotmob or {}).items() if k in known}
        self.w_fotmob = fotmob_weight(len(self.fotmob))

        # xG-Schätzung: Tore ≈ a·(Schüsse aufs Tor ohne Elfmeter) + b·(übrige Schüsse) + 0,76·Elfmeter
        X, Y = [], []
        for m, s in sm:
            for side, g in (("h", m.hg), ("a", m.ag)):
                t = s[side]
                X.append([max(t["aufs_tor"] - t["elfmeter"], 0), max(t["schuesse"] - t["aufs_tor"], 0)])
                Y.append(g - PENALTY_XG * t["elfmeter"])
        if len(X) >= 50:
            coef, *_ = np.linalg.lstsq(np.array(X, float), np.array(Y, float), rcond=None)
            self.xg_coef = (max(float(coef[0]), 0.05), max(float(coef[1]), 0.0))
        else:
            self.xg_coef = (0.30, 0.035)

        # Raumdaten: nur Spiele mit verlässlichen Positionen
        self.events = {m.id: s["ev"] for m, s in sm if s.get("ev") and s.get("evv") == EV_VERSION and events_ok(s["ev"])}
        self.shot_events = {m.id: s["ev"] for m, s in sm if s.get("ev") and s.get("evv") == EV_VERSION
                            and sum(e[1] == "S" for e in s["ev"]) >= 4}
        self.spatial = Spatial({str(k): v for k, v in self.events.items()},
                               {str(k): v for k, v in self.shot_events.items()})
        self.n_spatial = len(self.events)
        self.match_spatial: dict[int, dict] = {k: self.spatial.match_values(v) for k, v in self.events.items()}

        self.xg: dict[int, tuple[float, float]] = {}
        self.xg_source: dict[int, str] = {}
        self.hist: dict[int, list] = {}
        for m, s in sm:
            h, a = s["h"], s["a"]
            mv = self.match_spatial.get(m.id)
            if mv is not None:
                xh, xa = mv["xg"]["h"], mv["xg"]["a"]
                self.xg_source[m.id] = "Positionen"
                xt_d = mv["xt"]["h"] - mv["xt"]["a"]
                fk_h = 0.5 * mv["territorium_heim"] + 0.5 * h["ballbesitz"]
            elif m.id in self.shot_events:
                sx = {"h": 0.0, "a": 0.0}
                for e in self.shot_events[m.id]:
                    if e[1] == "S":
                        sx[e[0]] += self.spatial.shot_xg(e)
                k = XG_SCHUSS_ANTEIL
                xh, xa = k * sx["h"] + (1 - k) * self.xg_of(h), k * sx["a"] + (1 - k) * self.xg_of(a)
                self.xg_source[m.id] = "Schüsse mit Position"
                xt_d, fk_h = None, None
            else:
                xh, xa = self.xg_of(h), self.xg_of(a)
                self.xg_source[m.id] = "Schusszahlen"
                xt_d, fk_h = None, None
            if m.id in self.fotmob:
                fh, fa = self.fotmob[m.id]
                xh, xa = self.w_fotmob * fh + (1 - self.w_fotmob) * xh, self.w_fotmob * fa + (1 - self.w_fotmob) * xa
            self.xg[m.id] = (xh, xa)
            ppda_h = a["paesse"] / max(h["tacklings"] + h["ballgewinne"] + h["fouls"], 1)
            ppda_a = h["paesse"] / max(a["tacklings"] + a["ballgewinne"] + a["fouls"], 1)
            self.hist.setdefault(m.home_id, []).append(
                (m.kickoff, [xh - xa, h["schuesse"] - a["schuesse"], h["ballbesitz"] - 50, ppda_h,
                             xt_d, None if fk_h is None else fk_h - 50]))
            self.hist.setdefault(m.away_id, []).append(
                (m.kickoff, [xa - xh, a["schuesse"] - h["schuesse"], a["ballbesitz"] - 50, ppda_a,
                             None if xt_d is None else -xt_d, None if fk_h is None else 50 - fk_h]))
        for mid, (fh, fa) in self.fotmob.items():
            self.xg.setdefault(mid, (fh, fa))
        self.times = {t: [k for k, _ in v] for t, v in self.hist.items()}

        # Kader-Historie je Team (für Ausfälle von Schlüsselspielern)
        self.lineups: dict[int, list] = {}
        for m, s in sm:
            lu = s.get("lu") or {}
            for team, side in ((m.home_id, "h"), (m.away_id, "a")):
                if lu.get(side):
                    self.lineups.setdefault(team, []).append((m.kickoff, m.season, m.id, lu[side]))
        self.lu_times = {t: [k for k, *_ in v] for t, v in self.lineups.items()}
        self._abs_cache: dict = {}

        # Direkte Duelle (alle Ligen in den Daten), je Vereinspaar zeitlich sortiert
        self.duels: dict[frozenset, list[Match]] = {}
        for mm in sorted((x for x in matches if x.finished), key=lambda x: x.kickoff):
            self.duels.setdefault(frozenset((mm.home_id, mm.away_id)), []).append(mm)
        self._h2h_cache: dict = {}
        self.absence_mode = "vorher"     # "vorher": nur vor Anpfiff Bekanntes; "aufstellung": echte Startelf (nur Test)

        self.season_matches: dict[int, list[Match]] = {}
        for m in bl1:
            self.season_matches.setdefault(m.season, []).append(m)
        self._table_cache: dict = {}
        self._diff_cache: dict = {}

        # Skalen zur Standardisierung (aus allen Spielen vor Anpfiff berechnet)
        samples = {k: [] for k in FORM_WEIGHTS}
        for m in bl1:
            for t in (m.home_id, m.away_id):
                r = self.raw(t, m.kickoff, m.season)
                for k, v in r.items():
                    if v is not None and k in FORM_WEIGHTS:
                        samples[k].append(v)
        self.scale = {k: (float(np.mean(v)) if v else 0.0, float(np.std(v)) if len(v) > 5 else 1.0)
                      for k, v in samples.items()}

    def absence(self, team: int, m: Match) -> dict:
        """Anteil der Stärke, der durch vermutlich fehlende Schlüsselspieler wegfällt (0 = niemand fehlt)."""
        key = (team, m.id, self.absence_mode)
        if key in self._abs_cache:
            return self._abs_cache[key]
        res = {"wert": 0.0, "fehlen": []}
        hist = self.lineups.get(team, [])
        i = bisect.bisect_left(self.lu_times.get(team, []), m.kickoff)
        prev = hist[max(0, i - AUSFALL_SPIELE):i]
        if len(prev) >= 4:
            n = len(prev)
            starts, ga, names = {}, {}, {}
            for _, _, _, rows in prev:
                for pid, name, st, _sub, g, a, _y, _r in rows:
                    names[pid] = name
                    if st:
                        starts[pid] = starts.get(pid, 0) + 1
                    ga[pid] = ga.get(pid, 0) + g + a
            imp = {pid: (c / n) * (1 + min(ga.get(pid, 0) / c, 1.0)) for pid, c in starts.items() if c / n >= 0.5}
            keys = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:11])
            total = sum(keys.values()) or 1.0
            out = set()
            if self.absence_mode == "aufstellung":
                cur = next((x for x in hist[i:i + 1] if x[2] == m.id), None)
                if cur:
                    xi = {r[0] for r in cur[3] if r[2]}
                    out = {pid for pid in keys if pid not in xi}
            else:
                _, last_season, _, last_rows = prev[-1]
                in_squad = {r[0] for r in last_rows}
                season_y: dict[str, int] = {}
                for _, ssn, _, rows in hist[:i]:
                    if ssn == last_season:
                        for r in rows:
                            season_y[r[0]] = season_y.get(r[0], 0) + r[6]
                for r in last_rows:
                    pid = r[0]
                    if pid in keys and (r[7] > 0 or (r[6] > 0 and season_y.get(pid, 0) in (5, 10, 15))):
                        out.add(pid)          # Rote Karte oder Gelbsperre
                for pid in keys:
                    if pid not in in_squad:
                        out.add(pid)          # nicht im Kader des letzten Spiels (verletzt/krank)
            res = {"wert": sum(keys[p] for p in out) / total, "fehlen": sorted((names.get(p, p) for p in out))}
        self._abs_cache[key] = res
        return res

    def h2h(self, m: Match) -> dict:
        """Letzte direkte Duelle vor dem Spiel aus Sicht der Heimelf: heimvorteil-bereinigte
        Tordifferenz, neuere Duelle stärker gewichtet, bei wenigen Duellen zur 0 geschrumpft."""
        if m.id in self._h2h_cache:
            return self._h2h_cache[m.id]
        alle = self.duels.get(frozenset((m.home_id, m.away_id)), [])
        i = bisect.bisect_left([x.kickoff for x in alle], m.kickoff)
        letzte = alle[max(0, i - H2H_SPIELE):i][::-1]
        num = den = 0.0
        liste = []
        for k, x in enumerate(letzte):
            gd = (x.hg - x.ag) if x.home_id == m.home_id else (x.ag - x.hg)
            venue = H2H_HEIM if x.home_id == m.home_id else -H2H_HEIM
            w = H2H_ABKLINGEN ** k
            num += w * (gd - venue)
            den += w
            liste.append({"datum": x.kickoff.astimezone().strftime("%d.%m.%Y"), "heim": x.home, "gast": x.away,
                          "ergebnis": f"{x.hg}:{x.ag}", "liga": x.league})
        wert = (num / den) * (len(letzte) / H2H_SPIELE) if den else 0.0
        bil = {"s": 0, "u": 0, "n": 0}
        for x in letzte:
            gd = (x.hg - x.ag) if x.home_id == m.home_id else (x.ag - x.hg)
            bil["s" if gd > 0 else "u" if gd == 0 else "n"] += 1
        res = {"wert": wert, "duelle": liste, "bilanz": bil}
        self._h2h_cache[m.id] = res
        return res

    def h2h_diff(self, m: Match) -> float:
        return self.h2h(m)["wert"] if m.league == "bl1" else 0.0

    def absence_diff(self, m: Match) -> float:
        if m.league != "bl1":
            return 0.0
        return self.absence(m.away_id, m)["wert"] - self.absence(m.home_id, m)["wert"]

    def xg_of(self, t: dict) -> float:
        a, b = self.xg_coef
        return (a * max(t["aufs_tor"] - t["elfmeter"], 0) + b * max(t["schuesse"] - t["aufs_tor"], 0)
                + PENALTY_XG * t["elfmeter"])

    def table(self, season: int, before: datetime):
        key = (season, before.date())
        if key not in self._table_cache:
            pts, gd, gp = {}, {}, {}
            for m in self.season_matches.get(season, []):
                if m.kickoff.date() >= before.date():
                    break
                for t, gf, ga in ((m.home_id, m.hg, m.ag), (m.away_id, m.ag, m.hg)):
                    pts[t] = pts.get(t, 0) + (3 if gf > ga else 1 if gf == ga else 0)
                    gd[t] = gd.get(t, 0) + gf - ga
                    gp[t] = gp.get(t, 0) + 1
            order = sorted(pts, key=lambda t: (-pts[t], -gd[t]))
            self._table_cache[key] = ({t: i + 1 for i, t in enumerate(order)}, gp)
        return self._table_cache[key]

    def raw(self, team: int, before: datetime, season: int) -> dict:
        out = {k: None for k in FORM_WEIGHTS}
        ts = self.times.get(team, [])
        i = bisect.bisect_left(ts, before)
        recs = [v for _, v in self.hist.get(team, [])[max(0, i - FORM_GAMES):i]]
        out["n"] = len(recs)
        out["n_raum"] = 0
        if recs:
            arr = np.array([r[:4] for r in recs], float).mean(axis=0)
            out["xg"], out["schuesse"], out["ballbesitz"], out["ppda"] = (float(v) for v in arr)
            xt = [r[4] for r in recs if r[4] is not None]
            fk = [r[5] for r in recs if r[5] is not None]
            out["xt"] = float(np.mean(xt)) if xt else None
            out["feldkontrolle"] = float(np.mean(fk)) if fk else None
            out["n_raum"] = len(xt)
        ranks, gp = self.table(season, before)
        if team in ranks and gp.get(team, 0) >= 3:
            out["tabelle"] = (len(ranks) + 1) / 2 - ranks[team]
        return out

    def form(self, team: int, before: datetime, season: int) -> dict:
        r = self.raw(team, before, season)
        shrink = min(r["n"], FORM_GAMES) / FORM_GAMES
        z = {}
        for k in FORM_WEIGHTS:
            if r[k] is None:
                z[k] = 0.0
                continue
            mean, sd = self.scale[k]
            v = (r[k] - mean) / (sd or 1.0)
            if k == "ppda":
                v = -v            # wenige gegnerische Pässe pro Defensivaktion = starkes Pressing
            if k in ("xt", "feldkontrolle"):
                v *= min(r["n_raum"], FORM_GAMES) / FORM_GAMES
            elif k != "tabelle":
                v *= shrink
            z[k] = v
        return {"index": sum(FORM_WEIGHTS[k] * z[k] for k in FORM_WEIGHTS), "z": z, "raw": r}

    def diff_vec(self, m: Match) -> np.ndarray:
        """Differenz Heim minus Gast je Form-Faktor (standardisiert), Reihenfolge wie FORM_WEIGHTS."""
        if m.league != "bl1":
            return np.zeros(len(FORM_WEIGHTS))
        if m.id not in self._diff_cache:
            zh = self.form(m.home_id, m.kickoff, m.season)["z"]
            za = self.form(m.away_id, m.kickoff, m.season)["z"]
            self._diff_cache[m.id] = np.array([zh[k] - za[k] for k in FORM_WEIGHTS])
        return self._diff_cache[m.id]

    def diff(self, m: Match) -> float:
        return float(self.diff_vec(m) @ np.array(list(FORM_WEIGHTS.values())))


# --------------------------------------------------------------------------
# Modell
# --------------------------------------------------------------------------

HEIM_ABWEHR = True   # eigene Abwehrstärke je Team für Heimspiele (getrennte Heim-/Auswärtsleistung)
REG_HOME = 150.0    # Glättung der teamspezifischen Heimwerte (mit Heim-Abwehr und Historie ab 2021/22 im Test am besten)
REG_GAMMA = 2.0     # Regularisierung des Form-Effekts
RHO_GEWICHT = 1.0   # volle Dixon-Coles-Korrektur für knappe Ergebnisse (im Test realistischer als 0,5)
AUSFAELLE = True    # Ausfälle von Schlüsselspielern als eigener, aus den Daten gelernter Effekt
DUELLE = False      # direkte Duelle nur anzeigen: im Test an Saison 2025/26 verschlechterten sie die Prognose (+0,0017 Log-Loss)
H2H_SPIELE = 5      # so viele letzte Duelle zählen
H2H_ABKLINGEN = 0.75  # jedes ältere Duell zählt 25 % weniger als das neuere
H2H_HEIM = 0.35     # Heimvorteil in Toren, der aus jedem Duell herausgerechnet wird
FORM_MODUS = "fest"  # "fest": Form-Gewichte wie FORM_WEIGHTS, "lernen": je Faktor aus den Daten geschätzt
DRAW_FAKTOR = 1.10  # Unentschieden etwas häufiger als die reine Poisson-Rechnung (im Test kalibriert)
SCHAERFE = 1.2      # Siegchancen schärfen: das Modell war bei klaren Favoriten zu vorsichtig
TIPP_NAEHE = 0.7    # Tipp: unter Ergebnissen mit mind. 70 % der besten Wahrscheinlichkeit das den erwarteten Toren nächste
TIPP_SPREIZUNG = 1.25  # Tipp: erwarteter Torabstand × Faktor – klare Favoriten bekommen deutlichere Ergebnisse (im Test nah am echten Abstand, ohne messbaren Verlust)
TIPP_REMIS = 1.2    # nur für die Tipp-Auswahl: Unentschieden-Faktor (Prozente bleiben bei DRAW_FAKTOR); im Test mehr exakte Treffer


def _tau(x, y, lam, mu, rho):
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return np.maximum(t, 1e-10)


class DixonColes:
    """Dixon-Coles mit
    * Zielgröße = Mischung aus Toren und xG (Gewicht wxg),
    * teamspezifischem Heimvorteil (aus den eigenen Heimspielen, zeitgewichtet),
    * Form-Effekt gamma auf die Differenz der Form-Indizes beider Teams."""

    def __init__(self, xi: float, reg: float, wxg: float = 0.0, form_mode: str | None = None,
                 reg_gamma: float | None = None, rho_w: float | None = None, draw_f: float | None = None):
        self.xi, self.reg, self.wxg = xi, reg, wxg
        self.form_mode = form_mode or FORM_MODUS
        self.reg_gamma = REG_GAMMA if reg_gamma is None else reg_gamma
        self.rho_w = RHO_GEWICHT if rho_w is None else rho_w
        self.draw_f = DRAW_FAKTOR if draw_f is None else draw_f
        self.schaerfe = SCHAERFE
        self.use_aus = AUSFAELLE
        self.use_h2h = DUELLE

    def _dvec(self, m: Match) -> np.ndarray:
        if self.feats is None:
            return np.zeros(self.K)
        v = self.feats.diff_vec(m) if self.form_mode == "lernen" else np.array([self.feats.diff(m)])
        if self.use_aus:
            v = np.append(v, self.feats.absence_diff(m))
        if self.use_h2h:
            v = np.append(v, self.feats.h2h_diff(m))
        return v

    def fit(self, matches: list[Match], ref_time: datetime, feats: Features | None = None) -> "DixonColes":
        self.feats = feats
        ms = [m for m in matches if m.finished and m.kickoff < ref_time]
        teams = sorted({m.home_id for m in ms} | {m.away_id for m in ms})
        self.idx = {t: i for i, t in enumerate(teams)}
        self.names = {}
        for m in ms:
            self.names[m.home_id], self.names[m.away_id] = m.home, m.away
        n = len(teams)
        nl = len(LEAGUES)
        h = np.array([self.idx[m.home_id] for m in ms])
        a = np.array([self.idx[m.away_id] for m in ms])
        lg = np.array([LEAGUES.index(m.league) for m in ms])
        x = np.array([m.hg for m in ms], dtype=float)
        y = np.array([m.ag for m in ms], dtype=float)
        xe, ye = x.copy(), y.copy()
        self.feats = feats
        self.K = (len(FORM_WEIGHTS) if self.form_mode == "lernen" else 1) + (1 if self.use_aus else 0) + (1 if self.use_h2h else 0)
        K = self.K
        D = np.zeros((len(ms), K))
        if feats is not None:
            for k, m in enumerate(ms):
                if m.id in feats.xg and self.wxg > 0:
                    xh, xa = feats.xg[m.id]
                    xe[k] = self.wxg * xh + (1 - self.wxg) * m.hg
                    ye[k] = self.wxg * xa + (1 - self.wxg) * m.ag
                D[k] = self._dvec(m)
        days = np.array([(ref_time - m.kickoff).total_seconds() / 86400 for m in ms])
        w = np.exp(-self.xi * days)
        w = w / w.mean()
        reg, rg = self.reg, self.reg_gamma
        nh = n if HEIM_ABWEHR else 0
        o = 3 * n + nh + nl

        # Parameter: [att(n), def(n), heim_team(n), heim_abwehr(nh), liga(nl), heim, gamma(K)]
        def unpack(p):
            return p[:n], p[n:2 * n], p[2 * n:3 * n], p[3 * n:3 * n + nh], p[3 * n + nh:o], p[o], p[o + 1:]

        def nll(p):
            att, dfn, hadv, hdef, li, home, gam = unpack(p)
            d = D @ gam
            ll = li[lg] + home + hadv[h] + att[h] + dfn[a] + d / 2
            lm = li[lg] + att[a] + dfn[h] - d / 2
            if nh:
                lm = lm + hdef[h]
            lam, mu = np.exp(ll), np.exp(lm)
            f = (np.sum(w * (lam - xe * ll + mu - ye * lm)) + reg * (att @ att + dfn @ dfn)
                 + REG_HOME * (hadv @ hadv + hdef @ hdef) + rg * gam @ gam)
            gl, gm = w * (lam - xe), w * (mu - ye)
            g_att = np.bincount(h, gl, n) + np.bincount(a, gm, n) + 2 * reg * att
            g_def = np.bincount(a, gl, n) + np.bincount(h, gm, n) + 2 * reg * dfn
            g_had = np.bincount(h, gl, n) + 2 * REG_HOME * hadv
            g_hdf = (np.bincount(h, gm, n) + 2 * REG_HOME * hdef) if nh else np.zeros(0)
            g_li = np.bincount(lg, gl + gm, nl)
            g_gam = D.T @ (gl - gm) / 2 + 2 * rg * gam
            return f, np.concatenate([g_att, g_def, g_had, g_hdf, g_li, [gl.sum()], g_gam])

        p0 = np.zeros(o + 1 + K)
        p0[3 * n + nh:o] = math.log(1.4)
        p0[o] = 0.25
        res = minimize(nll, p0, jac=True, method="L-BFGS-B")
        self.att, self.dfn, self.hadv, hdef, self.li, self.home, gam = unpack(res.x)
        self.hdef = hdef if nh else np.zeros(n)
        self.gammas = np.array(gam, float)
        self.gamma = float(self.gammas[0]) if self.form_mode != "lernen" else float(np.abs(self.gammas[:len(FORM_WEIGHTS)]).sum())
        nf = len(FORM_WEIGHTS) if self.form_mode == "lernen" else 1
        self.gamma_aus = float(self.gammas[nf]) if self.use_aus else 0.0
        self.gamma_h2h = float(self.gammas[-1]) if self.use_h2h else 0.0

        d = D @ self.gammas
        ll = self.li[lg] + self.home + self.hadv[h] + self.att[h] + self.dfn[a] + d / 2
        lm = self.li[lg] + self.att[a] + self.dfn[h] + self.hdef[h] - d / 2
        lam, mu = np.exp(ll), np.exp(lm)
        r = minimize_scalar(lambda rho: -np.sum(w * np.log(_tau(x, y, lam, mu, rho))),
                            bounds=(-0.25, 0.25), method="bounded")
        self.rho = float(r.x)
        self.n_matches = len(ms)
        return self

    def _p(self, arr, team):
        i = self.idx.get(team)
        return float(arr[i]) if i is not None else 0.0

    def home_factor(self, team: int) -> float:
        return math.exp(self.home + self._p(self.hadv, team))

    def expected_goals(self, m: Match):
        li = self.li[LEAGUES.index(m.league)]
        d = float(self._dvec(m) @ self.gammas) if self.feats is not None else 0.0
        lam = (li + self.home + self._p(self.hadv, m.home_id) + self._p(self.att, m.home_id)
               + self._p(self.dfn, m.away_id) + d / 2)
        mu = li + self._p(self.att, m.away_id) + self._p(self.dfn, m.home_id) + self._p(self.hdef, m.home_id) - d / 2
        return math.exp(lam), math.exp(mu)

    def score_matrix(self, m: Match) -> np.ndarray:
        lam, mu = self.expected_goals(m)
        g = np.arange(MAX_GOALS + 1)
        mat = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        r = self.rho * self.rho_w
        mat[0, 0] *= 1 - lam * mu * r
        mat[0, 1] *= 1 + lam * r
        mat[1, 0] *= 1 + mu * r
        mat[1, 1] *= 1 - r
        mat = np.clip(mat, 0, None)
        if self.draw_f != 1.0:
            mat[np.diag_indices_from(mat)] *= self.draw_f
        mat = mat / mat.sum()
        if self.schaerfe != 1.0:
            # Sieg/Unentschieden/Niederlage-Wahrscheinlichkeiten p -> p^a (normiert), Ergebnisse darin anteilig
            g = np.arange(MAX_GOALS + 1)
            masks = [g[:, None] > g[None, :], g[:, None] == g[None, :], g[:, None] < g[None, :]]
            p = np.array([mat[mk].sum() for mk in masks])
            q = p ** self.schaerfe
            q = q / q.sum()
            for mk, pi, qi in zip(masks, p, q):
                if pi > 0:
                    mat[mk] *= qi / pi
        return mat

    def predict(self, m: Match) -> dict:
        mat = self.score_matrix(m)
        lam, mu = self.expected_goals(m)
        i, j = np.unravel_index(np.argmax(mat), mat.shape)
        flat = sorted(((mat[a, b], a, b) for a in range(MAX_GOALS + 1)
                       for b in range(MAX_GOALS + 1)), reverse=True)
        # bestes Ergebnis innerhalb der wahrscheinlichsten Tendenz (für Tippspiele)
        ph, pd, pa = np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()
        g = np.arange(MAX_GOALS + 1)
        tm = mat
        if TIPP_REMIS != self.draw_f:
            old, self.draw_f = self.draw_f, TIPP_REMIS
            tm = self.score_matrix(m)
            self.draw_f = old
        tp = [np.tril(tm, -1).sum(), np.trace(tm), np.triu(tm, 1).sum()]
        mask = [g[:, None] > g[None, :], g[:, None] == g[None, :], g[:, None] < g[None, :]][int(np.argmax(tp))]
        inmask = np.where(mask, tm, -1.0)
        best = inmask.max()
        ziel_d, ziel_s = TIPP_SPREIZUNG * (lam - mu), lam + mu
        cand = [((a - b - ziel_d) ** 2 + (a + b - ziel_s) ** 2, -tm[a, b], a, b)
                for a in range(MAX_GOALS + 1) for b in range(MAX_GOALS + 1)
                if mask[a, b] and tm[a, b] >= TIPP_NAEHE * best]
        cand.sort()
        _, _, ti, tj = cand[0]
        # weitere Kandidaten derselben Tendenz (für die Begrenzung gleicher Tipps je Spieltag)
        rest = sorted(((-tm[a, b], a, b) for a in range(MAX_GOALS + 1) for b in range(MAX_GOALS + 1)
                       if mask[a, b] and (a, b) not in {(c[2], c[3]) for c in cand}))
        kand = [[int(a), int(b), float(tm[a, b])] for _, _, a, b in cand] + \
               [[int(a), int(b), float(-q)] for q, a, b in rest]
        out = {
            "tipp": [int(ti), int(tj)],
            "tipp_kandidaten": kand[:8],
            "home": float(ph), "draw": float(pd), "away": float(pa),
            "score": [int(i), int(j)],
            "score_p": float(mat[i, j]),
            "top_scores": [[int(a), int(b), float(p)] for p, a, b in flat[:3]],
            # hoher Sieg: mit mindestens 3 Toren Abstand (im Test gut kalibriert: Favoriten > 70 % → 38 % vorhergesagt, 37 % eingetreten)
            "hoch": [float(mat[g[:, None] - g[None, :] >= 3].sum()), float(mat[g[None, :] - g[:, None] >= 3].sum())],
            "xg": [lam, mu],
            "heimfaktor": self.home_factor(m.home_id),
            "form": [0.0, 0.0],
        }
        out["ausfall"] = [{"wert": 0.0, "fehlen": []}, {"wert": 0.0, "fehlen": []}]
        if self.feats is not None and m.league == "bl1":
            out["form"] = [self.feats.form(m.home_id, m.kickoff, m.season)["index"],
                           self.feats.form(m.away_id, m.kickoff, m.season)["index"]]
            out["ausfall"] = [self.feats.absence(m.home_id, m), self.feats.absence(m.away_id, m)]
            out["h2h"] = self.feats.h2h(m)
        return out


# --------------------------------------------------------------------------
# Lernen: Backtest & Hyperparameter-Tuning
# --------------------------------------------------------------------------

MAX_GLEICHE_TIPPS = 6   # pro Spieltag höchstens so oft dasselbe Tipp-Ergebnis


def begrenze_tipps(preds: list[dict], max_gleich: int = MAX_GLEICHE_TIPPS) -> int:
    """Sorgt dafür, dass innerhalb eines Spieltags kein Tipp-Ergebnis öfter als max_gleich vorkommt.
    Überzählige Spiele bekommen ihren nächstbesten Tipp derselben Tendenz – jeweils dort, wo der
    Wechsel am wenigsten Wahrscheinlichkeit kostet. Gibt die Zahl der Änderungen zurück."""
    changes = 0
    while True:
        count: dict[tuple, int] = {}
        for p in preds:
            count[tuple(p["tipp"])] = count.get(tuple(p["tipp"]), 0) + 1
        over = [k for k, c in count.items() if c > max_gleich]
        if not over:
            return changes
        best = None
        for p in preds:
            if tuple(p["tipp"]) not in over:
                continue
            cur = next((c[2] for c in p.get("tipp_kandidaten", []) if [c[0], c[1]] == p["tipp"]), None)
            for a, b, q in p.get("tipp_kandidaten", []):
                if [a, b] != p["tipp"] and count.get((a, b), 0) < max_gleich:
                    loss = (cur if cur is not None else q) - q
                    if best is None or loss < best[0]:
                        best = (loss, p, [a, b])
                    break
        if best is None:
            return changes
        best[1]["tipp"] = best[2]
        changes += 1


MEISTER_SIMULATIONEN = 10000   # Durchläufe der Restsaison
MEISTER_STREUUNG = 0.05         # Unsicherheit der Teamstärke je Durchlauf; im Test (4 Saisons) bester Kompromiss aus Log-Loss und Brier


def simuliere_saison(model: "DixonColes", matches: list[Match], season: int, league: str = "bl1",
                     n: int = MEISTER_SIMULATIONEN, sigma: float | None = None, seed: int = 0,
                     methode: str = "poisson") -> dict:
    """Spielt die Restsaison n-mal zufällig durch (Tore je Spiel aus dem Modell) und zählt, wie oft
    jedes Team wo landet. Tabellenregeln: Punkte, Tordifferenz, erzielte Tore, sonst Los.
    sigma > 0 verschiebt je Durchlauf die Stärke jedes Teams zufällig (Unsicherheit des Modells)."""
    sigma = MEISTER_STREUUNG if sigma is None else sigma
    ms = [m for m in matches if m.league == league and m.season == season]
    teams = sorted({m.home_id for m in ms} | {m.away_id for m in ms})
    names = {}
    for m in ms:
        names[m.home_id], names[m.away_id] = m.home, m.away
    ix = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    pts, gd, gf = np.zeros(T), np.zeros(T), np.zeros(T)
    for m in ms:
        if m.finished:
            h, a = ix[m.home_id], ix[m.away_id]
            pts[h] += 3 if m.hg > m.ag else 1 if m.hg == m.ag else 0
            pts[a] += 3 if m.ag > m.hg else 1 if m.hg == m.ag else 0
            gd[h] += m.hg - m.ag; gd[a] += m.ag - m.hg
            gf[h] += m.hg; gf[a] += m.ag
    rest = [m for m in ms if not m.finished]
    rng = np.random.default_rng(seed)
    P, D, F = np.tile(pts, (n, 1)), np.tile(gd, (n, 1)), np.tile(gf, (n, 1))
    if rest:
        hi = np.array([ix[m.home_id] for m in rest]); ai = np.array([ix[m.away_id] for m in rest])
        if methode == "matrix":
            G1 = np.empty((n, len(rest)), int); G2 = np.empty((n, len(rest)), int)
            for k, m in enumerate(rest):
                cum = np.cumsum(model.score_matrix(m).ravel())
                cell = np.minimum(np.searchsorted(cum / cum[-1], rng.random(n)), cum.size - 1)
                G1[:, k], G2[:, k] = np.divmod(cell, MAX_GOALS + 1)
        else:
            eg = np.array([model.expected_goals(m) for m in rest])
            lam, mu = np.tile(eg[:, 0], (n, 1)), np.tile(eg[:, 1], (n, 1))
            if sigma > 0:
                d = rng.normal(0.0, sigma, (n, T))
                shift = np.exp(d[:, hi] - d[:, ai])
                lam, mu = lam * shift, mu / shift
            G1, G2 = rng.poisson(lam), rng.poisson(mu)
        oh = np.zeros((len(rest), T)); oh[np.arange(len(rest)), hi] = 1
        oa = np.zeros((len(rest), T)); oa[np.arange(len(rest)), ai] = 1
        ph = 3 * (G1 > G2) + (G1 == G2); pa = 3 * (G2 > G1) + (G1 == G2)
        P += ph @ oh + pa @ oa
        D += (G1 - G2) @ oh + (G2 - G1) @ oa
        F += G1 @ oh + G2 @ oa
    key = P * 1e6 + (D + 500) * 1e3 + F + rng.random((n, T)) * 0.5
    order = np.argsort(-key, axis=1)
    rank = np.empty_like(order); rank[np.arange(n)[:, None], order] = np.arange(T)
    platz_p = np.stack([np.bincount(rank[:, i], minlength=T) for i in range(T)]) / n
    return {"teams": [names[t] for t in teams], "team_ids": teams, "platz_p": platz_p,
            "meister": platz_p[:, 0], "erw_punkte": P.mean(0), "erw_platz": (rank + 1).mean(0),
            "punkte_jetzt": pts, "rest_spiele": len(rest)}


def _outcome(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def backtest(matches: list[Match], xi: float, reg: float, n_days: int,
             feats: Features | None = None, wxg: float = 0.0) -> dict:
    """Walk-forward: Für jeden der letzten n fertigen bl1-Spieltage wird nur mit
    Daten VOR dem Spieltag trainiert und dann die Bundesliga-Spiele der folgenden
    4 Tage vorhergesagt – genau wie im echten Einsatz."""
    done = [m for m in matches if m.finished]
    groups: dict[tuple, list[Match]] = {}
    for m in done:
        if m.league == "bl1":
            groups.setdefault((m.season, m.matchday), []).append(m)
    starts = sorted(min(mm.kickoff for mm in g) for g in groups.values())[-n_days:]
    ll = br = 0.0
    hits = exact = cnt = 0
    for start in starts:
        end = start + timedelta(days=4)
        day = [m for m in done if start <= m.kickoff < end and m.league == "bl1"]
        model = DixonColes(xi, reg, wxg).fit(matches, start, feats)
        for m in day:
            p = model.predict(m)
            probs = [p["home"], p["draw"], p["away"]]
            o = _outcome(m.hg, m.ag)
            ll -= math.log(max(probs[o], 1e-12))
            br += sum((probs[i] - (i == o)) ** 2 for i in range(3))
            hits += int(np.argmax(probs) == o)
            exact += int(p["score"] == [m.hg, m.ag])
            cnt += 1
    return {"logloss": ll / cnt, "brier": br / cnt, "tendenz": hits / cnt,
            "exakt": exact / cnt, "spiele": cnt}


def tune(matches: list[Match], feats: Features, verbose: bool = True) -> dict:
    best = None
    for wxg in WXG_GRID:
        for xi in XI_GRID:
            for reg in REG_GRID:
                r = backtest(matches, xi, reg, BACKTEST_MATCHDAYS, feats, wxg)
                if verbose:
                    print(f"  xG-Gewicht={wxg:.2f} xi={xi:.4f} reg={reg:<4} → LogLoss {r['logloss']:.4f}  "
                          f"Tendenz {r['tendenz']:.1%}  exakt {r['exakt']:.1%}")
                if best is None or r["logloss"] < best[0]["logloss"]:
                    best = (r, xi, reg, wxg)
    r, xi, reg, wxg = best
    if verbose:
        base = backtest(matches, xi, reg, BACKTEST_MATCHDAYS, None, 0.0)
        print(f"  Vergleich ohne xG und Form: LogLoss {base['logloss']:.4f}  "
              f"Tendenz {base['tendenz']:.1%}")
    return {"xi": xi, "reg": reg, "wxg": wxg, "backtest": r}


# --------------------------------------------------------------------------
# Zustand (Parameter + Vorhersage-Protokoll)
# --------------------------------------------------------------------------

def _load(name, default):
    f = DATA_DIR / name
    return json.loads(f.read_text("utf-8")) if f.exists() else default


def _save(name, obj):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False), "utf-8")


def evaluate_log(log: dict, matches: list[Match]) -> dict:
    """Trägt Ergebnisse in gespeicherte Vorhersagen ein und berechnet die Bilanz."""
    by_id = {m.id: m for m in matches}
    for mid, e in log.items():
        m = by_id.get(int(mid))
        if m and m.finished:
            e["result"] = [m.hg, m.ag]
    ev = [e for e in log.values() if e.get("result")]
    if not ev:
        return {"spiele": 0}
    hits = exact = 0
    ll = 0.0
    for e in ev:
        probs = [e["home"], e["draw"], e["away"]]
        o = _outcome(*e["result"])
        hits += int(np.argmax(probs) == o)
        exact += int(e["score"] == e["result"])
        ll -= math.log(max(probs[o], 1e-12))
    return {"spiele": len(ev), "tendenz": hits / len(ev), "exakt": exact / len(ev),
            "logloss": ll / len(ev)}


# --------------------------------------------------------------------------
# Ausgabe
# --------------------------------------------------------------------------

def _bar(p, width=20):
    n = round(p * width)
    return "█" * n + "·" * (width - n)


def print_predictions(title, rows):
    print(f"\n{title}\n" + "=" * len(title))
    for m, p in rows:
        ko = fmt_time(m.kickoff)
        live = "   (läuft bereits)" if m.kickoff <= datetime.now(timezone.utc) else ""
        print(f"\n{ko}   {m.home} – {m.away}{live}")
        print(f"  Heimsieg {m.home:<22} {p['home']:6.1%}  {_bar(p['home'])}")
        print(f"  Unentschieden{'':<18} {p['draw']:6.1%}  {_bar(p['draw'])}")
        print(f"  Auswärtssieg {m.away:<18} {p['away']:6.1%}  {_bar(p['away'])}")
        alt = ", ".join(f"{a}:{b} ({q:.0%})" for a, b, q in p["top_scores"][1:])
        print(f"  Wahrscheinlichstes Ergebnis: {p['score'][0]}:{p['score'][1]} "
              f"({p['score_p']:.1%})   danach: {alt}")
        print(f"  Tipp passend zur Tendenz:    {p['tipp'][0]}:{p['tipp'][1]}")
        print(f"  Erwartete Tore: {p['xg'][0]:.2f} : {p['xg'][1]:.2f}   "
              f"Form-Index {p['form'][0]:+.2f} / {p['form'][1]:+.2f}   "
              f"Heimvorteil {m.home} ×{p['heimfaktor']:.2f}")


def print_results(done_rows):
    if not done_rows:
        return
    print("\nBereits gespielt – unser Tipp vor Anpfiff")
    for e in done_rows:
        o_t = int(np.argmax([e["home"], e["draw"], e["away"]]))
        ok = "✔" if o_t == _outcome(*e["result"]) else "✘"
        tp = e.get("tipp", e["score"])
        ex = "  (exakt!)" if tp == e["result"] else ""
        print(f"  {ok} {e['spiel']:<32} Tipp {tp[0]}:{tp[1]}  "
              f"Ergebnis {e['result'][0]}:{e['result'][1]}{ex}")


def write_html(path: Path, title, rows, stats, params, model, done_rows=()):
    def esc(s):
        return html.escape(str(s))

    cards = []
    for m, p in rows:
        ko = fmt_time(m.kickoff, year=True)
        fav = max(("home", m.home), ("draw", "Unentschieden"), ("away", m.away),
                  key=lambda t: p[t[0]])
        alt = " · ".join(f"{a}:{b} <small>{q:.0%}</small>" for a, b, q in p["top_scores"][1:])
        cards.append(f"""
<article class="card">
  <div class="ko">{esc(ko)}</div>
  <div class="teams"><span>{esc(m.home)}</span><span class="score">{p['score'][0]}:{p['score'][1]}</span><span>{esc(m.away)}</span></div>
  <div class="bar">
    <div class="h" style="width:{p['home']*100:.1f}%"></div><div class="d" style="width:{p['draw']*100:.1f}%"></div><div class="a" style="width:{p['away']*100:.1f}%"></div>
  </div>
  <div class="pct"><span>{p['home']:.0%} Heim</span><span>{p['draw']:.0%} X</span><span>{p['away']:.0%} Gast</span></div>
  <div class="meta">Tipp-Favorit: <b>{esc(fav[1])}</b> · Ergebnis {p['score'][0]}:{p['score'][1]} mit {p['score_p']:.1%} · danach {alt}<br>
  Tipp passend zur Tendenz: <b>{p['tipp'][0]}:{p['tipp'][1]}</b> · Erwartete Tore {p['xg'][0]:.2f} : {p['xg'][1]:.2f}</div>
</article>""")
    if done_rows:
        items = []
        for e in done_rows:
            o_t = int(np.argmax([e["home"], e["draw"], e["away"]]))
            ok = o_t == _outcome(*e["result"])
            items.append(f"<li class='{'ok' if ok else 'no'}'>{'✔' if ok else '✘'} {esc(e['spiel'])}: "
                         f"Tipp {e['score'][0]}:{e['score'][1]} · Ergebnis <b>{e['result'][0]}:{e['result'][1]}</b></li>")
        cards.append("<article class='card'><b>Bereits gespielt</b><ul class='res'>" + "".join(items) + "</ul></article>")
    if stats.get("spiele"):
        st = (f"Bisher ausgewertete Vorhersagen: <b>{stats['spiele']}</b> · "
              f"Tendenz richtig: <b>{stats['tendenz']:.1%}</b> · "
              f"exaktes Ergebnis: <b>{stats['exakt']:.1%}</b>")
    else:
        st = "Noch keine eigenen Vorhersagen ausgewertet – nach dem Spieltag erneut starten."
    bt = params.get("backtest")
    btxt = (f"Backtest der letzten {BACKTEST_MATCHDAYS} Spieltage ({bt['spiele']} Spiele): "
            f"Tendenz {bt['tendenz']:.1%}, exakt {bt['exakt']:.1%}" if bt else "")
    doc = f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Bundesliga-Tipp</title>
<style>
:root{{--bg:#f4f5f7;--card:#fff;--fg:#1b1f24;--mute:#667;--h:#1f7a4d;--d:#9aa3ad;--a:#c0392b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111418;--card:#1b2026;--fg:#e8eaed;--mute:#9aa;--d:#5b636c}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}}
main{{max-width:760px;margin:0 auto;padding:24px 16px}}
h1{{margin:0 0 4px;font-size:24px}} .sub{{color:var(--mute);margin-bottom:20px}}
.card{{background:var(--card);border-radius:12px;padding:16px;margin-bottom:12px;box-shadow:0 1px 3px #0002}}
.ko{{color:var(--mute);font-size:13px}}
.teams{{display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:center;font-weight:600;font-size:17px;margin:6px 0 10px}}
.teams span:last-child{{text-align:right}} .score{{font-size:22px;font-variant-numeric:tabular-nums}}
.bar{{display:flex;height:10px;border-radius:5px;overflow:hidden}}
.h{{background:var(--h)}} .d{{background:var(--d)}} .a{{background:var(--a)}}
.pct{{display:flex;justify-content:space-between;font-size:13px;margin-top:4px;font-variant-numeric:tabular-nums}}
.meta{{color:var(--mute);font-size:13px;margin-top:8px}}
.res{{margin:8px 0 0;padding-left:0;list-style:none}} .res li{{padding:2px 0}} .ok{{color:var(--h)}} .no{{color:var(--a)}}
footer{{color:var(--mute);font-size:13px;margin-top:20px}}
</style></head><body><main>
<h1>{esc(title)}</h1>
<div class="sub">Erstellt {datetime.now().strftime('%d.%m.%Y %H:%M')} · trainiert mit {model.n_matches} Spielen</div>
{''.join(cards)}
<footer>{st}<br>{btxt}<br>Modell: Dixon-Coles-Poisson, Zeitgewichtung xi={params['xi']}, Regularisierung {params['reg']}, Heimvorteil ×{math.exp(model.home):.2f}, rho={model.rho:.3f}. Daten: OpenLigaDB.</footer>
</main></body></html>"""
    path.write_text(doc, "utf-8")


def team_rows(model: "DixonColes", feats: "Features", matches: list[Match], season: int, now: datetime):
    ids = {m.home_id for m in matches if m.league == "bl1" and m.season == season}
    ranks, _ = feats.table(season, now + timedelta(days=1))
    # Tabelle: Spiele, Siege/Unentschieden/Niederlagen, Tore, Punkte, letzte 5
    tab: dict[int, dict] = {t: {"sp": 0, "s": 0, "u": 0, "n": 0, "tore": 0, "gegentore": 0, "punkte": 0, "serie": []}
                            for t in ids}
    for m in sorted((m for m in matches if m.league == "bl1" and m.season == season and m.finished),
                    key=lambda m: m.kickoff):
        for t, gf, ga in ((m.home_id, m.hg, m.ag), (m.away_id, m.ag, m.hg)):
            if t not in tab:
                continue
            r = tab[t]
            r["sp"] += 1
            r["tore"] += gf
            r["gegentore"] += ga
            k = "s" if gf > ga else "u" if gf == ga else "n"
            r[k] += 1
            r["punkte"] += {"s": 3, "u": 1, "n": 0}[k]
            r["serie"].append(k.upper())
    st = []
    for t in ids:
        if t not in model.idx:
            continue
        i = model.idx[t]
        f = feats.form(t, now, season)
        r = f["raw"]

        def rd(v, k=2):
            return None if v is None else round(v, k)
        st.append({"team": model.names[t], "angriff": round(math.exp(model.att[i]), 2),
                   "abwehr": round(math.exp(model.dfn[i]), 2),
                   "staerke": round(float(model.att[i] - model.dfn[i]), 3),
                   "heimfaktor": round(model.home_factor(t), 2),
                   "form": round(f["index"], 2), "form_spiele": r["n"],
                   "xg_diff": rd(r["xg"]), "schuss_diff": rd(r["schuesse"], 1),
                   "ballbesitz": rd(None if r["ballbesitz"] is None else r["ballbesitz"] + 50, 1),
                   "ppda": rd(r["ppda"], 1), "platz": ranks.get(t),
                   "xt_diff": rd(r["xt"], 3),
                   "feldkontrolle": rd(None if r["feldkontrolle"] is None else r["feldkontrolle"] + 50, 1),
                   **{k: v for k, v in tab[t].items() if k != "serie"},
                   "tordifferenz": tab[t]["tore"] - tab[t]["gegentore"],
                   "letzte5": "".join(tab[t]["serie"][-5:])})
    st.sort(key=lambda r: -r["staerke"])
    for k, r in enumerate(st, 1):
        r["rang"] = k
    return st


def print_table(model: DixonColes, matches: list[Match], season: int, feats=None, now=None):
    if feats is not None:
        st = team_rows(model, feats, matches, season, now)
        print("\nTeamstärken (1.0 = Ligadurchschnitt), Form = letzte 6 Spiele")
        print(f"{'#':>2}  {'Team':<16} {'Angr.':>6} {'Abw.*':>6} {'Heim':>5} {'Form':>6} "
              f"{'xG±':>6} {'Sch±':>6} {'Ball%':>6} {'PPDA':>5} {'Pl.':>4}")
        f = lambda v, fmt: ("–" if v is None else format(v, fmt))
        for r in st:
            print(f"{r['rang']:>2}  {r['team']:<16} {r['angriff']:6.2f} {r['abwehr']:6.2f} "
                  f"{r['heimfaktor']:5.2f} {r['form']:+6.2f} {f(r['xg_diff'], '+6.2f'):>6} "
                  f"{f(r['schuss_diff'], '+6.1f'):>6} {f(r['ballbesitz'], '6.1f'):>6} "
                  f"{f(r['ppda'], '5.1f'):>5} {f(r['platz'], '4d'):>4}")
        print("* Abwehr: Faktor für Gegentore – kleiner ist besser. Heim: eigener Heimvorteil-Faktor.")
        return
    ids = {m.home_id for m in matches if m.league == "bl1" and m.season == season}
    rows = []
    for t in ids:
        if t in model.idx:
            i = model.idx[t]
            rows.append((model.att[i] - model.dfn[i], model.names[t],
                         math.exp(model.att[i]), math.exp(model.dfn[i])))
    rows.sort(reverse=True)
    print("\nTeamstärken (1.0 = Ligadurchschnitt)")
    print(f"{'#':>2}  {'Team':<22} {'Angriff':>8} {'Abwehr*':>8}")
    for k, (_, n, at, df) in enumerate(rows, 1):
        print(f"{k:>2}  {n:<22} {at:8.2f} {df:8.2f}")
    print("* Abwehr: Faktor für Gegentore – kleiner ist besser")


# --------------------------------------------------------------------------
# Dashboard-Export
# --------------------------------------------------------------------------

def _pct(x):
    return round(100 * x, 1)


def import_protokoll(path: Path) -> dict:
    """Liest eine exportierte protokoll.json zurück in das Vorhersage-Protokoll."""
    rows = json.loads(path.read_text("utf-8"))["rows"]
    log = {}
    for r in rows:
        e = {"spiel": r["spiel"], "liga": r["liga"], "saison": r["saison"],
             "spieltag": r["spieltag"], "anpfiff": r["anpfiff"], "erstellt": r["erstellt"],
             "home": r["p_heim"] / 100, "draw": r["p_remis"] / 100, "away": r["p_gast"] / 100,
             "score": [int(v) for v in r["prognose"].split(":")],
             "tipp": [int(v) for v in r["tipp"].split(":")],
             "xg": [r["xg_heim"], r["xg_gast"]], "art": r.get("art", "live")}
        if r.get("ergebnis"):
            e["result"] = [int(v) for v in r["ergebnis"].split(":")]
        log[str(r["match_id"])] = e
    return log


def import_ereignisse(path: Path) -> dict:
    """Liest eine exportierte ereignisse.json zurück (Ereignisse je Spiel, ohne berechnetes xG)."""
    out = {}
    for r in json.loads(path.read_text("utf-8"))["rows"]:
        out[str(r["match_id"])] = {"ev": [e[:7] for e in json.loads(r["ereignisse"])],
                                   "lu": json.loads(r.get("aufstellung") or "{}"), "evv": r.get("ev_version")}
    return out


def import_statistik(path: Path) -> dict:
    """Liest eine exportierte statistik.json zurück in den Statistik-Zwischenspeicher."""
    out = {}
    for r in json.loads(path.read_text("utf-8"))["rows"]:
        out[str(r["match_id"])] = {s: {k: float(r[f"{k}_{n}"]) for k in STAT_FIELDS}
                                   for s, n in (("h", "heim"), ("a", "gast"))}
    return out


def export_dashboard(out: Path, season, day, rows, log, params, model, matches, now, feats=None, stats=None):
    out.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).read_text("utf-8")
    stamp = now.isoformat(timespec="seconds")
    meta = [{"label": "bundesliga_tipp.py", "value": code, "type": "code", "lang": "python"},
            {"label": "Daten", "value": "Ergebnisse: OpenLigaDB (api.openligadb.de), 1. und 2. Bundesliga, "
                                         f"Saisons {season - SEASONS_BACK}–{season}. Schüsse, Ballbesitz, Pässe, "
                                         f"Defensivaktionen: ESPN (site.api.espn.com), 1. Bundesliga, "
                                         f"Saisons {season - STAT_SEASONS_BACK}–{season}. xG aus Schüssen geschätzt."},
            {"label": "Berechnet", "value": stamp}]

    def write(name, data):
        (out / name).write_text(json.dumps({"rows": data, "meta": meta}, ensure_ascii=False), "utf-8")

    t_now = now
    write("prognosen.json", [{
        "match_id": m.id, "spieltag": m.matchday, "anpfiff": m.kickoff.isoformat(),
        "heim": m.home, "gast": m.away,
        "p_heim": _pct(p["home"]), "p_remis": _pct(p["draw"]), "p_gast": _pct(p["away"]),
        "p_hoch_heim": _pct(p["hoch"][0]), "p_hoch_gast": _pct(p["hoch"][1]),
        "prognose": f"{p['score'][0]}:{p['score'][1]}", "prognose_p": _pct(p["score_p"]),
        "tipp": f"{p['tipp'][0]}:{p['tipp'][1]}",
        "alternativen": ", ".join(f"{a}:{b}" for a, b, _ in p["top_scores"][1:]),
        "xg_heim": round(p["xg"][0], 2), "xg_gast": round(p["xg"][1], 2),
        "form_heim": round(p["form"][0], 2), "form_gast": round(p["form"][1], 2),
        "heimfaktor": round(p["heimfaktor"], 2),
        "ausfall_heim": round(100 * p["ausfall"][0]["wert"], 1), "ausfall_gast": round(100 * p["ausfall"][1]["wert"], 1),
        "fehlen_heim": ", ".join(p["ausfall"][0]["fehlen"]), "fehlen_gast": ", ".join(p["ausfall"][1]["fehlen"]),
        "duell_index": round(p.get("h2h", {}).get("wert", 0.0), 2),
        "duell_bilanz": "{s}-{u}-{n}".format(**p["h2h"]["bilanz"]) if p.get("h2h") else "",
        "duelle": json.dumps(p.get("h2h", {}).get("duelle", []), ensure_ascii=False),
        "status": "läuft" if m.kickoff <= t_now else "offen",
    } for m, p in rows])

    prot = []
    for mid, e in sorted(log.items(), key=lambda kv: kv[1].get("anpfiff", "")):
        res = e.get("result")
        probs = [e["home"], e["draw"], e["away"]]
        tend_ok = int(np.argmax(probs)) == _outcome(*res) if res else None
        score = e["score"]
        tipp = e.get("tipp", score)
        prot.append({
            "match_id": int(mid), "liga": e.get("liga", "bl1"), "saison": e.get("saison", season),
            "spieltag": e.get("spieltag"), "anpfiff": e.get("anpfiff"), "erstellt": e.get("erstellt"),
            "spiel": e["spiel"], "art": e.get("art", "live"),
            "p_heim": _pct(e["home"]), "p_remis": _pct(e["draw"]), "p_gast": _pct(e["away"]),
            "prognose": f"{score[0]}:{score[1]}", "tipp": f"{tipp[0]}:{tipp[1]}",
            "xg_heim": round(e.get("xg", [0, 0])[0], 2), "xg_gast": round(e.get("xg", [0, 0])[1], 2),
            "ergebnis": f"{res[0]}:{res[1]}" if res else None,
            "ausgewertet": 1 if res else 0,
            "tendenz_richtig": (1 if tend_ok else 0) if res else None,
            "exakt": (1 if score == res else 0) if res else None,
            "tipp_exakt": (1 if tipp == res else 0) if res else None,
        })
    write("protokoll.json", prot)

    staerken = team_rows(model, feats, matches, season, now)
    sim = simuliere_saison(model, matches, season)
    sim_by = {n: i for i, n in enumerate(sim["teams"])}
    for r in staerken:
        i = sim_by.get(r["team"])
        if i is not None:
            r["meister_pct"] = round(100 * float(sim["meister"][i]), 2)
            r["erw_punkte"] = round(float(sim["erw_punkte"][i]), 1)
            r["erw_platz"] = round(float(sim["erw_platz"][i]), 1)
    write("staerken.json", staerken)

    by_id = {m.id: m for m in matches}
    srows = []
    for mid, s in sorted(((k, v) for k, v in stats.items() if int(k) in by_id),
                         key=lambda kv: by_id[int(kv[0])].kickoff):
        m = by_id[int(mid)]
        r = {"match_id": m.id, "saison": m.season, "spieltag": m.matchday, "anpfiff": m.kickoff.isoformat(),
             "heim": m.home, "gast": m.away, "tore_heim": m.hg, "tore_gast": m.ag}
        for side, name in (("h", "heim"), ("a", "gast")):
            for k in STAT_FIELDS:
                r[f"{k}_{name}"] = s[side][k]
        xg = feats.xg.get(m.id)
        r["xg_heim"], r["xg_gast"] = (round(xg[0], 2), round(xg[1], 2)) if xg else (None, None)
        fm = feats.fotmob.get(m.id)
        r["xg_fotmob_heim"], r["xg_fotmob_gast"] = fm if fm else (None, None)
        r["xg_quelle"] = feats.xg_source.get(m.id)
        mv = feats.match_spatial.get(m.id)
        r["xt_heim"], r["xt_gast"] = (round(mv["xt"]["h"], 3), round(mv["xt"]["a"], 3)) if mv else (None, None)
        r["feldkontrolle_heim"] = round(0.5 * mv["territorium_heim"] + 0.5 * s["h"]["ballbesitz"], 1) if mv else None
        srows.append(r)
    write("statistik.json", srows)

    # Ereignisse mit Position (für Schusskarte und für den nächsten Lauf)
    erows = []
    for mid, s in sorted(((k, v) for k, v in stats.items() if int(k) in by_id and v.get("ev")),
                         key=lambda kv: by_id[int(kv[0])].kickoff):
        m = by_id[int(mid)]
        ok = m.id in feats.events
        ok_xg = m.id in feats.shot_events
        ev = [e + [round(feats.spatial.shot_xg(e), 3) if (e[1] == "S" and ok_xg) else None] for e in s["ev"]]
        erows.append({"match_id": m.id, "saison": m.season, "spieltag": m.matchday, "heim": m.home, "gast": m.away,
                      "tore_heim": m.hg, "tore_gast": m.ag, "positionen_ok": ok, "ev_version": s.get("evv"), "ereignisse": json.dumps(ev),
                      "aufstellung": json.dumps(s.get("lu", {}), ensure_ascii=False)})
    write("ereignisse.json", erows)

    bt = params.get("backtest", {})
    write("modell.json", [{
        "saison": f"{season}/{(season + 1) % 100:02d}", "spieltag": day,
        "berechnet": stamp, "trainiert_spiele": model.n_matches,
        "heimvorteil": round(math.exp(model.home), 2), "rho": round(model.rho, 3),
        "xi": params["xi"], "reg": params["reg"],
        "xg_gewicht": _pct(params["wxg"]), "form_effekt": round(model.gamma, 3),
        "form_effekt_pct": round(100 * (math.exp(model.gamma) - 1), 1),
        "gewichte": ", ".join(f"{FORM_LABELS[k]} {_pct(v):g} %" for k, v in FORM_WEIGHTS.items()),
        "xg_je_schuss_aufs_tor": round(feats.xg_coef[0], 3),
        "xg_je_sonstigem_schuss": round(feats.xg_coef[1], 3),
        "statistik_spiele": feats.n_stat_matches,
        "raum_spiele": feats.n_spatial, "raum_schuesse": feats.spatial.n_shots,
        "elfmeter_xg": round(feats.spatial.penalty_xg, 2),
        "rho_gewicht": _pct(RHO_GEWICHT), "unentschieden_faktor": DRAW_FAKTOR, "schaerfe": SCHAERFE,
        "form_spiele": FORM_GAMES,
        "meister_simulationen": MEISTER_SIMULATIONEN, "meister_streuung": MEISTER_STREUUNG,
        "duell_effekt_pct": round(100 * (math.exp(abs(getattr(model, "gamma_h2h", 0.0))) - 1), 1) if DUELLE else None,
        "ausfall_effekt_pct": round(100 * (math.exp(abs(getattr(model, "gamma_aus", 0.0)) * 0.25) - 1), 1) if AUSFAELLE else None,
        "fotmob_spiele": len(feats.fotmob),
        "fotmob_gewicht": _pct(feats.w_fotmob),
        "fotmob_gewicht_start": _pct(FOTMOB_MIN), "fotmob_gewicht_max": _pct(FOTMOB_MAX),
        "fotmob_voll_ab": FOTMOB_FULL,
        "halbwertszeit_tage": round(math.log(2) / params["xi"]),
        "backtest_spiele": bt.get("spiele"),
        "backtest_tendenz": _pct(bt["tendenz"]) if bt else None,
        "backtest_exakt": _pct(bt["exakt"]) if bt else None,
    }])


# --------------------------------------------------------------------------
# Daten für die öffentliche Website (GitHub Pages)
# --------------------------------------------------------------------------

SITE_FILES = ["prognosen", "protokoll", "staerken", "modell", "statistik", "ereignisse"]


def write_site_data(site_dir: Path, export_dir: Path) -> list[str]:
    """Schreibt schlanke Kopien der Export-Dateien nach <site_dir>/data: ohne eingebetteten
    Quelltext, bei den Ereignissen nur Spiele mit genauen Positionen (für die Schusskarte).
    Eine Datei wird nur geschrieben, wenn sich ihre Zeilen geändert haben."""
    changed = []
    out = site_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    for name in SITE_FILES:
        src = export_dir / f"{name}.json"
        if not src.exists():
            continue
        d = json.loads(src.read_text("utf-8"))
        rows = d["rows"]
        if name == "ereignisse":
            rows = [{k: v for k, v in r.items() if k != "aufstellung"} for r in rows if r.get("positionen_ok")]
        meta = [m for m in d.get("meta", []) if m.get("type") != "code"]
        target = out / f"{name}.json"
        if target.exists():
            try:
                if json.loads(target.read_text("utf-8")).get("rows") == rows:
                    continue
            except ValueError:
                pass
        target.write_text(json.dumps({"rows": rows, "meta": meta}, ensure_ascii=False, separators=(",", ":")), "utf-8")
        changed.append(f"data/{name}.json")
    return changed


# --------------------------------------------------------------------------
# Hauptprogramm
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Bundesliga-Vorhersagen")
    ap.add_argument("--spieltag", type=int, help="bestimmten Spieltag der laufenden Saison vorhersagen")
    ap.add_argument("--liga", default="bl1", choices=LEAGUES, help="bl1 (Standard) oder bl2")
    ap.add_argument("--tune", action="store_true", help="Hyperparameter jetzt neu optimieren")
    ap.add_argument("--tabelle", action="store_true", help="Teamstärken anzeigen")
    ap.add_argument("--offline", action="store_true", help="nur gespeicherte Daten verwenden")
    ap.add_argument("--html", default="vorhersage.html", help="Pfad für den HTML-Bericht")
    ap.add_argument("--export", metavar="ORDNER", help="Dashboard-Daten (JSON) in diesen Ordner schreiben")
    ap.add_argument("--rueckblick", action="store_true",
                    help="gespielte Spieltage dieser Saison nachträglich (nur mit Daten vor Anpfiff) tippen")
    ap.add_argument("--import-protokoll", metavar="DATEI",
                    help="Vorhersage-Protokoll aus einer exportierten protokoll.json übernehmen")
    ap.add_argument("--import-fotmob", metavar="DATEI_ODER_ORDNER",
                    help="von Hand eingetragene FotMob-xG (aus dem Dashboard) verwenden")
    ap.add_argument("--site", metavar="ORDNER",
                    help="zusätzlich schlanke Daten für die Website nach ORDNER/data schreiben (braucht --export)")
    ap.add_argument("--import-ereignisse", metavar="DATEI",
                    help="Spielereignisse mit Position aus einer exportierten ereignisse.json übernehmen")
    ap.add_argument("--import-statistik", metavar="DATEI",
                    help="Spielstatistiken aus einer exportierten statistik.json übernehmen")
    args = ap.parse_args(argv)

    now = datetime.now(timezone.utc)
    season = current_season(now)
    print("Lade Spieldaten von OpenLigaDB …")
    matches = load_matches(now, args.offline)
    print("Lade Spielstatistiken (Schüsse, Ballbesitz, Pässe) von ESPN …")
    imported = import_statistik(Path(args.import_statistik)) if args.import_statistik else None
    if imported is not None and args.import_ereignisse:
        for k, e in import_ereignisse(Path(args.import_ereignisse)).items():
            if k in imported:
                imported[k]["ev"] = e["ev"]
                if e.get("evv") is not None:
                    imported[k]["evv"] = e["evv"]
                if e["lu"]:
                    imported[k]["lu"] = e["lu"]
    spielstat = load_stats(matches, now, args.offline, imported)
    fotmob = {}
    if args.import_fotmob:
        fotmob = load_fotmob(Path(args.import_fotmob))
        _save("fotmob_xg.json", {str(k): v for k, v in fotmob.items()})
    else:
        fotmob = {int(k): tuple(v) for k, v in _load("fotmob_xg.json", {}).items()}
    feats = Features(matches, spielstat, fotmob)
    if feats.fotmob:
        print(f"  FotMob-xG für {len(feats.fotmob)} Spiele, Gewicht {feats.w_fotmob:.0%}")
    print(f"  {feats.n_stat_matches} Spiele mit Statistik, davon {feats.n_spatial} mit genauen Positionen, "
          f"{len(feats.shot_events)} mit Schusspositionen ({feats.spatial.n_shots} Schüsse für das xG-Modell)")

    # Lernen: Hyperparameter neu optimieren, wenn neue Ergebnisse vorliegen
    params = _load("parameter.json", dict(DEFAULT_PARAMS))
    done_days = len({(m.season, m.matchday) for m in matches if m.finished and m.league == "bl1"})
    if args.tune or params.get("tuned_days") != done_days or params.get("version") != 6:
        print(f"Neue Ergebnisse gefunden – optimiere Modell per Backtest "
              f"(letzte {BACKTEST_MATCHDAYS} Spieltage) …")
        t = tune(matches, feats, verbose=args.tune)
        params.update(xi=t["xi"], reg=t["reg"], wxg=t["wxg"], backtest=t["backtest"],
                      tuned_days=done_days, version=6)
        _save("parameter.json", params)
        print(f"  beste Einstellung: xG-Gewicht={t['wxg']}, xi={t['xi']}, reg={t['reg']} "
              f"(Tendenz {t['backtest']['tendenz']:.1%} im Backtest)")

    model = DixonColes(params["xi"], params["reg"], params["wxg"]).fit(matches, now, feats)

    # Zu tippende Spiele bestimmen
    cur = [m for m in matches if m.league == args.liga and m.season == season]
    if args.spieltag:
        day = args.spieltag
    else:
        open_days = sorted({m.matchday for m in cur if not m.finished})
        if not open_days:
            print("Keine offenen Spiele in dieser Saison.")
            return
        day = open_days[0]
    day_matches = sorted([m for m in cur if m.matchday == day], key=lambda m: m.kickoff)
    target = [m for m in day_matches if not m.finished] or day_matches
    rows = [(m, model.predict(m)) for m in target]
    begrenze_tipps([p for _, p in rows])
    # Vorschau: alle weiteren Spieltage nach aktuellem Stand (je Spieltag eigene Begrenzung gleicher Tipps)
    alle_rows = list(rows)
    for d in sorted({m.matchday for m in cur if not m.finished and m.matchday > day}):
        dr = [(m, model.predict(m)) for m in sorted((m for m in cur if m.matchday == d and not m.finished),
                                                    key=lambda m: m.kickoff)]
        begrenze_tipps([p for _, p in dr])
        alle_rows += dr

    # Vorhersagen protokollieren (nur noch nicht gespielte Spiele, vor Anpfiff)
    log = _load("vorhersagen.json", {})
    if args.import_protokoll:
        log.update(import_protokoll(Path(args.import_protokoll)))
    if args.rueckblick:
        for d in sorted({m.matchday for m in cur if m.finished}):
            dm = [m for m in cur if m.matchday == d and m.finished and str(m.id) not in log]
            if not dm:
                continue
            start = min(m.kickoff for m in cur if m.matchday == d)
            bm = DixonColes(params["xi"], params["reg"], params["wxg"]).fit(matches, start, feats)
            preds = {m.id: bm.predict(m) for m in dm}
            begrenze_tipps(list(preds.values()))
            for m in dm:
                p = preds[m.id]
                log[str(m.id)] = {"spiel": f"{m.home} – {m.away}", "liga": m.league,
                                  "saison": season, "spieltag": d, "anpfiff": m.kickoff.isoformat(),
                                  "erstellt": start.isoformat(), "art": "rückblick",
                                  **{k: p[k] for k in ("home", "draw", "away", "score", "tipp", "xg")}}
    for m, p in rows:
        if not m.finished and m.kickoff > now:
            log[str(m.id)] = {"spiel": f"{m.home} – {m.away}", "liga": m.league,
                              "saison": season, "spieltag": day,
                              "anpfiff": m.kickoff.isoformat(), "erstellt": now.isoformat(),
                              "art": "live",
                              **{k: p[k] for k in ("home", "draw", "away", "score", "tipp", "xg")}}
    stats = evaluate_log(log, matches)
    _save("vorhersagen.json", log)
    done_rows = [log[str(m.id)] for m in day_matches
                 if m.finished and log.get(str(m.id), {}).get("result")]

    league_name = "Bundesliga" if args.liga == "bl1" else "2. Bundesliga"
    title = f"{league_name} {season}/{(season + 1) % 100:02d} – {day}. Spieltag"
    print_predictions(title, rows)
    print_results(done_rows)
    if args.tabelle:
        print_table(model, matches, season, feats, now)

    print(f"\nTrainiert mit {model.n_matches} Spielen, "
          f"Heimvorteil ×{math.exp(model.home):.2f}, rho={model.rho:.3f}")
    if stats.get("spiele"):
        print(f"Eigene Bilanz: {stats['spiele']} Spiele ausgewertet, Tendenz "
              f"{stats['tendenz']:.1%}, exakt {stats['exakt']:.1%}")
    out = Path(args.html).resolve()
    write_html(out, title, rows, stats, params, model, done_rows)
    print(f"HTML-Bericht: {out}")
    if args.export:
        export_dashboard(Path(args.export), season, day, alle_rows, log, params, model, matches, now, feats, spielstat)
        print(f"Dashboard-Daten: {Path(args.export).resolve()}")
        if args.site:
            ch = write_site_data(Path(args.site), Path(args.export))
            print("Website-Daten geändert: " + (", ".join(ch) if ch else "keine"))


if __name__ == "__main__":
    main()
