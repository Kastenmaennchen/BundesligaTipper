# Bundesliga-Tipp

Prognosen, Tabelle und Analysen zur Fußball-Bundesliga: Gewinnwahrscheinlichkeiten, wahrscheinlichste Ergebnisse, Tipp, Schusskarten mit xG, Gefahr (xT) und Feldkontrolle.

**Website:** über GitHub Pages dieses Repositorys (Settings → Pages → Branch `main`, Ordner `/ (root)`).

## Wie die Seite aktuell bleibt

Jede Nacht gegen 23:40 Uhr rechnet ein geplanter Lauf das Modell neu (Ergebnisse von OpenLigaDB, Spielstatistik und Spielereignisse von ESPN, von Hand eingetragene FotMob-xG) und legt die neuen Daten in `data/` ab. Die Seite lädt nur diese JSON-Dateien; es gibt keinen Server und keine Anmeldung.

## Inhalt

| Pfad | Inhalt |
|---|---|
| `index.html` | die Seite |
| `site.js` | lädt die Daten und zeichnet die Bereiche |
| `data/*.json` | Prognosen, Tipp-Protokoll, Tabelle und Teamwerte, Modell-Kennzahlen, Spielstatistik, Spielereignisse |
| `d3.min.js` | d3 v7.9.0 (ISC-Lizenz) |
| `fonts/` | Cormorant (SIL Open Font License 1.1) |

Die Datendateien enthalten im Feld `meta` die Datenquellen und den Zeitpunkt der Berechnung. Bei den Spielereignissen sind nur Spiele mit genauen Feldpositionen enthalten (für die Schusskarte).

Daten: [OpenLigaDB](https://www.openligadb.de) und ESPN. Alle Angaben ohne Gewähr.
