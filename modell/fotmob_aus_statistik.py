#!/usr/bin/env python3
"""Holt die bisher eingetragenen FotMob-xG aus data/statistik.json der Website zurück,
damit der nächtliche Lauf sie weiter berücksichtigt.

    python modell/fotmob_aus_statistik.py data/statistik.json fotmob.json
"""
import json
import sys
from pathlib import Path

quelle, ziel = Path(sys.argv[1]), Path(sys.argv[2])
rows = json.loads(quelle.read_text("utf-8"))["rows"] if quelle.exists() else []
eintraege = [{"match_id": r["match_id"], "xg_heim": r["xg_fotmob_heim"], "xg_gast": r["xg_fotmob_gast"]}
             for r in rows if r.get("xg_fotmob_heim") is not None and r.get("xg_fotmob_gast") is not None]
ziel.write_text(json.dumps(eintraege), "utf-8")
print(f"FotMob-xG übernommen: {len(eintraege)} Spiele")
