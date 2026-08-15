# Photo Face Sorter

Backend na automatické roztriedenie fotografií podľa tvárí. Nájde tváre na fotkách,
vygeneruje pre ne 512-rozmerné ArcFace embeddingy a nepodobné tváre zoskupí do osôb —
bez toho, aby ktokoľvek musel vopred označiť, kto je kto.

Beží plne lokálne na CPU, žiadne fotky neopúšťajú počítač.

## Ako to funguje

```
fotky/  ──►  FaceDetector  ──►  512d embeddingy  ──►  FaceClusterer  ──►  person_1, person_2, …
             (InsightFace)      + náhľady tvárí       (DBSCAN, cosine)     + unassigned
```

1. **Detekcia** — InsightFace (`buffalo_l`) nájde na fotke tváre, pre každú vygeneruje
   L2-normalizovaný 512d embedding a uloží orezaný náhľad do `CACHE_DIR/{face_id}.jpg`.
2. **Zhlukovanie** — DBSCAN s kosínusovou metrikou zoskupí embeddingy do osôb. Počet osôb
   sa nezadáva vopred; tváre, ktoré nikam nesadnú, skončia ako `unassigned`.
3. **Reprezentant** — pre každú osobu sa vyberie tvár najbližšie k centroidu zhluku, čiže
   tá „najtypickejšia", ktorá sa dá použiť ako náhľad osoby v UI.

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

```bash
python test_ml_pipeline.py "C:\fotky\dovolenka"
python test_ml_pipeline.py ./fotky --eps 0.45 --min-samples 3 --no-recursive -v
```

Ukážka výstupu:

```
DETEKCIA TVÁRÍ
Spracovaných fotografií : 7
Nájdených tvárí         : 18
Skóre detekcie          : min 0.871 / max 0.920 / priemer 0.891

ZHLUKOVANIE (DBSCAN)
Identifikovaných osôb   : 6
Nepriradených tvárí     : 0

OSOBA         TVÁRÍ  FOTIEK   NÁHĽAD
person_1          3       3   b357dce530eb4dbd8b4e351a7bcb87e2.jpg
person_2          3       3   6ae8a36154d7409ba1e0ea7fe35490f8.jpg
…
```

Ako knižnica:

```python
from app.services.detector import FaceDetector
from app.services.clusterer import FaceClusterer

faces, failures = FaceDetector().process_directory("C:/fotky")
result = FaceClusterer().cluster(faces)

for cluster in result.person_clusters:
    print(cluster.name, cluster.size, cluster.representative_preview_path)
```

## Konfigurácia

Nastavenia sa načítajú z prostredia alebo zo súboru `.env` (vzor v `.env.example`).

| Premenná | Default | Popis |
|---|---|---|
| `DBSCAN_EPS` | `0.5` | Max. kosínusová vzdialenosť dvoch tvárí tej istej osoby. Nižšia = prísnejšie. |
| `DBSCAN_MIN_SAMPLES` | `2` | Minimálny počet tvárí na vytvorenie osoby. |
| `CACHE_DIR` | `cache/faces` | Kam sa ukladajú orezané náhľady tvárí. |
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
  config.py              # Pydantic Settings (.env)
  schemas.py             # DetectedFace, FaceCluster, ClusteringResult
  services/
    detector.py          # FaceDetector — detekcia, embeddingy, náhľady
    clusterer.py         # FaceClusterer — DBSCAN, pomenovanie osôb
test_ml_pipeline.py      # CLI: detekcia + clustering nad priečinkom
```

## Poznámky k výkonu

- Na CPU zhruba **10 s/fotku** pri `det_size=640`. Pre väčšie dávky sa oplatí GPU
  (`INSIGHTFACE_CTX_ID=0` + `onnxruntime-gpu`) alebo nižší `det_size`.
- Obrázky sa načítavajú cez Pillow (nie `cv2.imread`) kvôli EXIF rotácii a cestám
  s diakritikou na Windows.
- Podporované formáty: JPG, JPEG, PNG, BMP, WEBP, TIF, TIFF.

## Tech stack

Python 3.11+ · InsightFace (ArcFace `buffalo_l`) · ONNX Runtime · scikit-learn · Pillow · Pydantic v2

## Licencia

MIT
