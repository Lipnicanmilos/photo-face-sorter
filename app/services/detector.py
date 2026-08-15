"""Detekcia tvárí a generovanie ArcFace embeddingov cez InsightFace."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import Settings, get_settings
from app.schemas import EMBEDDING_DIM, BoundingBox, DetectedFace

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)


class FaceDetectionError(RuntimeError):
    """Fotografiu sa nepodarilo načítať alebo spracovať."""


class FaceDetector:
    """Obal nad InsightFace `FaceAnalysis` - detekcia, embeddingy a náhľady tvárí.

    Model sa načítava lenivo (lazy) pri prvom použití, takže inštanciu detektora
    je možné vytvoriť aj tam, kde sa ešte nevie, či bude naozaj potrebná.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._model: object | None = None
        self._settings.ensure_directories()

    # ------------------------------------------------------------------ model

    @property
    def model(self) -> object:
        """Lenivo inicializovaný InsightFace model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> object:
        from insightface.app import FaceAnalysis  # lokálny import - ťažká závislosť

        cfg = self._settings
        providers = (
            ["CPUExecutionProvider"]
            if cfg.INSIGHTFACE_CTX_ID < 0
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        logger.info(
            "Načítavam InsightFace model '%s' (root=%s, providers=%s)",
            cfg.INSIGHTFACE_MODEL_NAME,
            cfg.INSIGHTFACE_ROOT,
            providers,
        )
        model = FaceAnalysis(
            name=cfg.INSIGHTFACE_MODEL_NAME,
            root=str(cfg.INSIGHTFACE_ROOT),
            providers=providers,
        )
        model.prepare(
            ctx_id=cfg.INSIGHTFACE_CTX_ID,
            det_size=(cfg.INSIGHTFACE_DET_SIZE, cfg.INSIGHTFACE_DET_SIZE),
        )
        return model

    def warmup(self) -> None:
        """Vynúti načítanie modelu dopredu (užitočné pri štarte servera)."""
        _ = self.model

    # -------------------------------------------------------------- verejné API

    def process_image(self, image_path: str) -> list[DetectedFace]:
        """Deteguje tváre na fotografii, vyrobí embeddingy a uloží náhľady.

        Args:
            image_path: Cesta k fotografii.

        Returns:
            Zoznam detegovaných tvárí zoradený od najväčšej po najmenšiu.

        Raises:
            FaceDetectionError: Ak sa obrázok nedá načítať alebo dekódovať.
        """
        path = Path(image_path)
        rgb_image = self._load_rgb_image(path)
        bgr_image = rgb_image[:, :, ::-1]  # InsightFace očakáva BGR (OpenCV konvenciu)

        raw_faces = self.model.get(bgr_image)  # type: ignore[attr-defined]
        cfg = self._settings
        detected: list[DetectedFace] = []

        for raw_face in raw_faces:
            bbox = self._to_bounding_box(raw_face.bbox, rgb_image.shape)
            det_score = float(getattr(raw_face, "det_score", 0.0))

            if det_score < cfg.MIN_DET_SCORE:
                logger.debug("Preskakujem tvár s nízkym skóre %.3f v %s", det_score, path)
                continue
            if bbox.width < cfg.MIN_FACE_SIZE or bbox.height < cfg.MIN_FACE_SIZE:
                logger.debug("Preskakujem príliš malú tvár %dx%d v %s", bbox.width, bbox.height, path)
                continue

            embedding = self._extract_embedding(raw_face)
            if embedding is None:
                logger.warning("Tvár bez embeddingu v %s - preskakujem", path)
                continue

            face_id = uuid.uuid4().hex
            preview_path = self._save_preview(rgb_image, bbox, face_id)

            detected.append(
                DetectedFace(
                    face_id=face_id,
                    source_path=str(path),
                    bbox=bbox,
                    det_score=min(det_score, 1.0),
                    preview_path=str(preview_path),
                    embedding=embedding,
                    age=self._optional_int(getattr(raw_face, "age", None)),
                    gender=self._decode_gender(getattr(raw_face, "gender", None)),
                )
            )

        detected.sort(key=lambda face: face.bbox.area, reverse=True)
        logger.info("%s: detegovaných %d tvárí", path.name, len(detected))
        return detected

    def process_directory(
        self,
        directory: str,
        recursive: bool = True,
    ) -> tuple[list[DetectedFace], list[tuple[str, str]]]:
        """Spracuje všetky podporované obrázky v adresári.

        Args:
            directory: Koreňový adresár s fotografiami.
            recursive: Či prehľadávať aj podadresáre.

        Returns:
            Dvojicu `(tváre, chyby)`, kde chyba je `(cesta, popis chyby)`.
        """
        faces: list[DetectedFace] = []
        failures: list[tuple[str, str]] = []

        for image_path in self.iter_images(directory, recursive=recursive):
            try:
                faces.extend(self.process_image(str(image_path)))
            except FaceDetectionError as exc:
                logger.warning("Preskakujem %s: %s", image_path, exc)
                failures.append((str(image_path), str(exc)))
            except Exception as exc:  # pragma: no cover - obrana pri chybe modelu
                logger.exception("Neočakávaná chyba pri spracovaní %s", image_path)
                failures.append((str(image_path), f"{type(exc).__name__}: {exc}"))

        return faces, failures

    @staticmethod
    def iter_images(directory: str, recursive: bool = True) -> Iterator[Path]:
        """Vráti zoradený iterátor podporovaných obrázkov v adresári."""
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(f"Adresár neexistuje: {root}")

        pattern = "**/*" if recursive else "*"
        candidates: Iterable[Path] = root.glob(pattern)
        yield from sorted(
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    # ---------------------------------------------------------------- pomocné

    @staticmethod
    def _load_rgb_image(path: Path) -> np.ndarray:
        """Načíta obrázok ako RGB pole, s rešpektovaním EXIF orientácie.

        Pillow sa používa zámerne namiesto `cv2.imread`, ktorý na Windows zlyháva
        pri cestách s diakritikou a ignoruje EXIF rotáciu.
        """
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                return np.asarray(image.convert("RGB"))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise FaceDetectionError(f"Obrázok sa nepodarilo načítať: {exc}") from exc

    @staticmethod
    def _to_bounding_box(bbox: np.ndarray, image_shape: tuple[int, ...]) -> BoundingBox:
        """Zaokrúhli a oreže bounding box na rozmery obrázka."""
        height, width = image_shape[0], image_shape[1]
        x1, y1, x2, y2 = (float(value) for value in bbox[:4])
        return BoundingBox(
            x1=int(np.clip(round(x1), 0, width)),
            y1=int(np.clip(round(y1), 0, height)),
            x2=int(np.clip(round(x2), 0, width)),
            y2=int(np.clip(round(y2), 0, height)),
        )

    @staticmethod
    def _extract_embedding(raw_face: object) -> np.ndarray | None:
        """Vráti L2-normalizovaný 512d embedding, alebo None ak chýba."""
        embedding = getattr(raw_face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(raw_face, "embedding", None)
        if embedding is None:
            return None

        vector = np.asarray(embedding, dtype=np.float32).ravel()
        if vector.size != EMBEDDING_DIM:
            logger.warning("Neočakávaná dimenzia embeddingu: %d", vector.size)
            return None

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return None
        return vector / norm

    def _save_preview(self, rgb_image: np.ndarray, bbox: BoundingBox, face_id: str) -> Path:
        """Oreže tvár s okrajom, zmenší ju a uloží ako CACHE_DIR/{face_id}.jpg."""
        cfg = self._settings
        height, width = rgb_image.shape[0], rgb_image.shape[1]

        margin_x = int(round(bbox.width * cfg.FACE_CROP_MARGIN))
        margin_y = int(round(bbox.height * cfg.FACE_CROP_MARGIN))
        left = max(0, bbox.x1 - margin_x)
        top = max(0, bbox.y1 - margin_y)
        right = min(width, bbox.x2 + margin_x)
        bottom = min(height, bbox.y2 + margin_y)

        crop = rgb_image[top:bottom, left:right]
        if crop.size == 0:  # degenerovaný box - použijeme aspoň pôvodný výrez
            crop = rgb_image[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        if crop.size == 0:
            raise FaceDetectionError(f"Prázdny výrez tváre {face_id}")

        preview = Image.fromarray(crop).resize(
            (cfg.PREVIEW_SIZE, cfg.PREVIEW_SIZE), Image.Resampling.LANCZOS
        )
        target = cfg.CACHE_DIR / f"{face_id}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        preview.save(target, format="JPEG", quality=cfg.PREVIEW_JPEG_QUALITY, optimize=True)
        return target

    @staticmethod
    def _optional_int(value: object) -> int | None:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _decode_gender(value: object) -> str | None:
        """InsightFace vracia 1 pre muža a 0 pre ženu."""
        try:
            return "M" if int(value) == 1 else "F"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
