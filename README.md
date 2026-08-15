# Photo Face Sorter

Backend na automatické roztriedenie fotografií podľa tvárí. Nájde tváre na fotkách,
vygeneruje pre ne 512-rozmerné ArcFace embeddingy a nepodobné tváre zoskupí do osôb —
bez toho, aby ktokoľvek musel vopred označiť, kto je kto.

Beží plne lokálne na CPU, žiadne fotky neopúšťajú počítač.

## Ako to funguje

```
fotky/ ──► FaceDetector ──► SQLite ──► FaceClusterer ──► PhotoOrganizer ──► output/Mama/
           (InsightFace)    (cache)    (DBSCAN)          (hardlinky)        output/person_2/
```

1. **Detekcia** — InsightFace (`buffalo_l`) nájde na fotke tváre, pre každú vygeneruje
   L2-normalizovaný 512d embedding a uloží orezaný náhľad do `CACHE_DIR/{face_id}.jpg`.
2. **Perzistencia** — fotka sa v SQLite identifikuje SHA-256 obsahu, takže pri ďalšom
   behu sa preskočí. Detekcia je najdrahšia časť (~10 s/fotku na CPU); druhý scan
   toho istého priečinka je otázka milisekúnd a model sa vôbec nenačíta.
3. **Zhlukovanie** — DBSCAN s kosínusovou metrikou zoskupí embeddingy do osôb. Počet osôb
   sa nezadáva vopred; tváre, ktoré nikam nesadnú, skončia ako `unassigned`.
4. **Triedenie** — fotky sa rozložia do `output/{osoba}/`. Fotka s piatimi ľuďmi patrí do
   piatich priečinkov, preto sa predvolene používajú **hardlinky** — na disku existuje
   stále len jeden fyzický súbor.

## Inštalácia

```bash
python -m venv venv
venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### Model buffalo_l (~325 MB)

InsightFace si model pri prvom spustení stiahne sám do `models/buffalo_l/`.
Ak sťahovanie zlyhá na SSL chybe (`CERTIFICATE_VERIFY_FAILED`), stiahni ho ručne:

```powershell
Invoke-WebRequest -Uri "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" -OutFile "models\buffalo_l.zip"
Expand-Archive "models\buffalo_l.zip" -DestinationPath "models\buffalo_l"
```

V `models/buffalo_l/` má byť 5 `.onnx` súborov (`det_10g`, `w600k_r50`, `2d106det`, `1k3d68`, `genderage`).

## Použitie

Všetko naraz — detekcia, zhlukovanie aj roztriedenie:

```bash
python -m app.cli run "C:\fotky\dovolenka" output
```

Alebo po krokoch:

```bash
python -m app.cli scan "C:\fotky"          # detekcia; známe fotky preskočí
python -m app.cli cluster --eps 0.45       # prepočet osôb
python -m app.cli sort output --dry-run    # ukáže plán bez zápisu
python -m app.cli sort output --clean      # roztriedi a upratie predošlý výstup
```

Pomenovanie osôb a prehľad:

```bash
python -m app.cli persons                  # zoznam osôb s počtom tvárí
python -m app.cli rename person_1 "Mama"   # priečinok sa bude volať Mama/
python -m app.cli stats                    # obsah databázy
python -m app.cli reset --yes              # vyprázdni databázu
```

Meno zadané cez `rename` **prežije aj ďalšie zhlukovanie** — prenesie sa na zhluk,
ktorý po prečíslovaní zdedil najviac pôvodných tvárí.

Ukážka výstupu:

```
SCAN
Fotiek v priečinku : 7
Novo spracovaných  : 1  (+0 tvárí)
Preskočených (cache): 6

ZHLUKOVANIE
Identifikovaných osôb : 6
Nepriradených tvárí   : 0

TRIEDENIE
PRIEČINOK                     FOTIEK
Mama                               3
Deda Jozef                         3
person_3                           3
_bez_tvari                         4
```

### Ako knižnica

```python
from app.pipeline import PhotoSorterPipeline

with PhotoSorterPipeline() as pipeline:
    pipeline.scan("C:/fotky")
    result = pipeline.recluster()
    pipeline.organize("output", mode="hardlink")

    for cluster in result.person_clusters:
        print(cluster.name, cluster.size, cluster.representative_preview_path)
```

Samotné ML časti sa dajú použiť aj bez databázy:

```python
from app.services.detector import FaceDetector
from app.services.clusterer import FaceClusterer

faces, failures = FaceDetector().process_directory("C:/fotky")
result = FaceClusterer().cluster(faces)
```

## Ako sa fotky triedia

```
output/
  Mama/            skupina_original.jpg  skupina_svetla.jpg  …
  Deda Jozef/      skupina_original.jpg  …          ← tá istá fotka, jeden hardlink
  person_3/        …
  _nezaradene/     fotky s tvárami, ktoré nesadli k žiadnej osobe
  _bez_tvari/      fotky bez detegovanej tváre
  .photo-face-sorter.json                ← manifest, podľa neho vie --clean upratať
```

Poistky pri zápise:

- Do priečinka s **cudzím obsahom** nástroj nič nezapíše — bez manifestu skončí chybou.
- `--clean` maže **iba priečinky uvedené v manifeste**, nie celý cieľový adresár.
- Zdrojové fotky sa nikdy nepresúvajú ani nemenia — vždy sa len linkujú/kopírujú.
- Ak hardlink zlyhá (iný disk, FAT32), automaticky sa použije kópia a nahlási sa to.

## Konfigurácia

Nastavenia sa načítajú z prostredia alebo zo súboru `.env` (vzor v `.env.example`).

| Premenná | Default | Popis |
|---|---|---|
| `DBSCAN_EPS` | `0.5` | Max. kosínusová vzdialenosť dvoch tvárí tej istej osoby. Nižšia = prísnejšie. |
| `DBSCAN_MIN_SAMPLES` | `2` | Minimálny počet tvárí na vytvorenie osoby. |
| `CACHE_DIR` | `cache/faces` | Kam sa ukladajú orezané náhľady tvárí. |
| `DB_PATH` | `cache/photo_face_sorter.sqlite3` | SQLite s fotkami, tvárami a osobami. |
| `OUTPUT_DIR` | `output` | Predvolený cieľ triedenia. |
| `LINK_MODE` | `hardlink` | `hardlink`, `copy` alebo `symlink`. |
| `INSIGHTFACE_MODEL_NAME` | `buffalo_l` | Balík modelov InsightFace. |
| `INSIGHTFACE_ROOT` | koreň projektu | Knižnica si doň sama vytvorí `models/{názov}`. |
| `INSIGHTFACE_CTX_ID` | `-1` | `-1` = CPU, `0+` = index GPU. |
| `INSIGHTFACE_DET_SIZE` | `640` | Rozlíšenie vstupu detektora. Menšie = rýchlejšie, menej malých tvárí. |
| `MIN_DET_SCORE` | `0.5` | Pod týmto skóre sa detekcia zahodí. |
| `MIN_FACE_SIZE` | `32` | Minimálna veľkosť tváre v pixeloch. |

### Ladenie `DBSCAN_EPS`

- Osoba sa rozpadla na viac zhlukov → **zvýš** `eps` (napr. 0.55).
- Dvaja ľudia splynuli do jedného → **zníž** `eps` (napr. 0.42).
- Veľa tvárí v `unassigned` → zníž `min_samples` na 2, prípadne zvýš `eps`.

## Štruktúra

```
app/
  cli.py                 # príkazový riadok (scan/cluster/sort/run/persons/…)
  pipeline.py            # PhotoSorterPipeline — spája detekciu, DB, clustering, triedenie
  db.py                  # SQLite: photos / faces / persons
  config.py              # Pydantic Settings (.env)
  schemas.py             # DetectedFace, FaceCluster, PersonRecord, reporty
  console.py             # UTF-8 výstup a formátovanie
  services/
    detector.py          # FaceDetector — detekcia, embeddingy, náhľady
    clusterer.py         # FaceClusterer — DBSCAN, pomenovanie osôb
    organizer.py         # PhotoOrganizer — triedenie do priečinkov, hardlinky
test_ml_pipeline.py      # samostatný test ML častí bez databázy
```

## Poznámky k výkonu

- Na CPU zhruba **10 s/fotku** pri `det_size=640`. Pre väčšie dávky sa oplatí GPU
  (`INSIGHTFACE_CTX_ID=0` + `onnxruntime-gpu`) alebo nižší `det_size`.
- Vďaka cache platíš tento čas len raz — opakovaný `scan` toho istého priečinka
  trvá milisekundy a model sa vôbec nenačíta.
- Obrázky sa načítavajú cez Pillow (nie `cv2.imread`) kvôli EXIF rotácii a cestám
  s diakritikou na Windows.
- Podporované formáty: JPG, JPEG, PNG, BMP, WEBP, TIF, TIFF.

## Tech stack

Python 3.11+ · InsightFace (ArcFace `buffalo_l`) · ONNX Runtime · scikit-learn · Pillow · Pydantic v2

## Licencia

MIT — detaily v súbore [LICENSE](LICENSE).
