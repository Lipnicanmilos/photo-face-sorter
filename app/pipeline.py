"""Prepojenie detekcie, perzistencie, zhlukovania a triedenia do jedného celku."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from app.config import Settings, get_settings
from app.db import Database, compute_file_hash
from app.schemas import ClusteringResult, OrganizeReport, ScanReport
from app.services.clusterer import FaceClusterer
from app.services.detector import FaceDetectionError, FaceDetector
from app.services.organizer import LinkMode, PhotoOrganizer

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, Path, bool], None]
"""(index, celkovo, cesta, bola_z_cache) - volané po každej fotke."""


class PhotoSorterPipeline:
    """Vysokoúrovňové API nad celým procesom triedenia fotiek podľa tvárí.

    Detektor sa načítava až vtedy, keď je naozaj čo detegovať - opakovaný scan
    už spracovaného priečinka tak beží v zlomku sekundy a model sa vôbec nedotkne.
    """

    def __init__(
        self,
        db: Database | None = None,
        settings: Settings | None = None,
        detector: FaceDetector | None = None,
        clusterer: FaceClusterer | None = None,
    ) -> None:
        cfg = settings or get_settings()
        cfg.ensure_directories()
        self._settings: Settings = cfg
        self.db: Database = db if db is not None else Database(settings=cfg)
        self._detector: FaceDetector | None = detector
        self.clusterer: FaceClusterer = clusterer or FaceClusterer(settings=cfg)

    @property
    def detector(self) -> FaceDetector:
        """Lenivo vytvorený detektor (model sa načíta až pri prvom použití)."""
        if self._detector is None:
            self._detector = FaceDetector(settings=self._settings)
        return self._detector

    # ------------------------------------------------------------------- scan

    def scan(
        self,
        folder: str | Path,
        recursive: bool = True,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> ScanReport:
        """Spracuje fotky v priečinku a uloží tváre do databázy.

        Fotka sa preskočí, ak je už v databáze s rovnakým SHA-256 obsahu - takže
        opakovaný beh nič nedetekuje znova a zmenené súbory sa prepočítajú.

        Args:
            folder: Priečinok s fotografiami.
            recursive: Prehľadávať aj podpriečinky.
            force: Spracovať aj fotky, ktoré sú už v databáze.
            progress: Voliteľný callback volaný po každej fotke.
        """
        started = time.perf_counter()
        images = list(FaceDetector.iter_images(str(folder), recursive=recursive))
        report = ScanReport(total_images=len(images))

        for index, image_path in enumerate(images, start=1):
            try:
                content_hash = compute_file_hash(image_path)
                file_size = image_path.stat().st_size
            except OSError as exc:
                report.failures.append((str(image_path), f"nedá sa čítať: {exc}"))
                continue

            cached = not force and self.db.is_photo_processed(image_path, content_hash)
            if cached:
                report.skipped_cached += 1
            else:
                try:
                    faces = self.detector.process_image(str(image_path))
                except FaceDetectionError as exc:
                    report.failures.append((str(image_path), str(exc)))
                    continue
                except Exception as exc:  # pragma: no cover - obrana pri chybe modelu
                    logger.exception("Neočakávaná chyba pri %s", image_path)
                    report.failures.append((str(image_path), f"{type(exc).__name__}: {exc}"))
                    continue

                self.db.save_photo_with_faces(image_path, content_hash, file_size, faces)
                report.processed += 1
                report.new_faces += len(faces)

            if progress is not None:
                progress(index, len(images), image_path, cached)

        report.elapsed_seconds = time.perf_counter() - started
        logger.info(
            "Scan hotový: %d nových, %d z cache, %d chýb za %.1f s",
            report.processed,
            report.skipped_cached,
            len(report.failures),
            report.elapsed_seconds,
        )
        return report

    # -------------------------------------------------------------- zhlukovanie

    def recluster(
        self,
        eps: float | None = None,
        min_samples: int | None = None,
    ) -> ClusteringResult:
        """Prepočíta osoby zo všetkých tvárí v databáze a uloží priradenia.

        Ručne zadané mená osôb zostávajú zachované - prenesú sa na zhluk, ktorý
        po prečíslovaní zdedil najviac pôvodných tvárí.
        """
        if eps is not None or min_samples is not None:
            self.clusterer = FaceClusterer(
                settings=self._settings, eps=eps, min_samples=min_samples
            )

        faces = self.db.load_faces()
        result = self.clusterer.cluster(faces)
        self.db.save_clustering(result)
        return result

    # ---------------------------------------------------------------- triedenie

    def organize(
        self,
        output_dir: str | Path | None = None,
        mode: LinkMode | None = None,
        dry_run: bool = False,
        clean: bool = False,
    ) -> OrganizeReport:
        """Roztriedi fotky do priečinkov podľa osôb uložených v databáze."""
        organizer = PhotoOrganizer(db=self.db, settings=self._settings)
        return organizer.organize(
            output_dir=Path(output_dir) if output_dir is not None else None,
            mode=mode,
            dry_run=dry_run,
            clean=clean,
        )

    # ------------------------------------------------------------- údržba

    def prune_missing(self) -> int:
        """Odstráni z databázy fotky, ktoré medzitým zmizli z disku."""
        return self.db.remove_missing_photos()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> PhotoSorterPipeline:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
