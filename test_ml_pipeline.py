"""Manuálny test celého ML pipeline-u: detekcia tvárí + zhlukovanie.

Použitie:
    python test_ml_pipeline.py <cesta_k_priecinku_s_fotkami> [--eps 0.5] [--min-samples 2]

Pri prvom spustení si InsightFace stiahne model buffalo_l (~300 MB), takže
inicializácia detektora trvá dlhšie než ďalšie behy.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from app.config import get_settings
from app.schemas import ClusteringResult, DetectedFace
from app.services.clusterer import FaceClusterer
from app.services.detector import FaceDetector

LINE_WIDTH: int = 72


def configure_console() -> None:
    """Prepne výstup na UTF-8, aby diakritika neskončila na cp1252 chybe (Windows)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prebehne detekciu tvárí a zhlukovanie nad priečinkom s fotografiami.",
    )
    parser.add_argument("folder", type=Path, help="Priečinok s fotografiami.")
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Prepíše DBSCAN_EPS z konfigurácie (kosínusová vzdialenosť).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Prepíše DBSCAN_MIN_SAMPLES z konfigurácie.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Neprehľadávať podpriečinky.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Podrobné logovanie (DEBUG).",
    )
    return parser.parse_args(argv)


def header(title: str) -> None:
    print()
    print("=" * LINE_WIDTH)
    print(title)
    print("=" * LINE_WIDTH)


def print_detection_stats(
    faces: list[DetectedFace],
    failures: list[tuple[str, str]],
    num_images: int,
    elapsed: float,
) -> None:
    header("DETEKCIA TVÁRÍ")
    print(f"Spracovaných fotografií : {num_images}")
    print(f"Nájdených tvárí         : {len(faces)}")
    print(f"Chybných súborov        : {len(failures)}")
    print(f"Čas                     : {elapsed:.2f} s", end="")
    if num_images:
        print(f"  ({elapsed / num_images:.2f} s/fotku)")
    else:
        print()

    if faces:
        scores = [face.det_score for face in faces]
        per_image = Counter(face.source_path for face in faces)
        print(f"Skóre detekcie          : min {min(scores):.3f} / max {max(scores):.3f} "
              f"/ priemer {sum(scores) / len(scores):.3f}")
        print(f"Tvárí na fotku          : max {max(per_image.values())} "
              f"/ priemer {len(faces) / len(per_image):.2f}")
        print(f"Náhľady uložené v       : {get_settings().CACHE_DIR}")

    for path, error in failures[:10]:
        print(f"  [chyba] {path}: {error}")
    if len(failures) > 10:
        print(f"  ... a ďalších {len(failures) - 10} chýb")


def print_clustering_stats(result: ClusteringResult, faces: list[DetectedFace], elapsed: float) -> None:
    clusterer_faces = {face.face_id: face for face in faces}

    header("ZHLUKOVANIE (DBSCAN)")
    print(f"Identifikovaných osôb   : {result.num_persons}")
    print(f"Nepriradených tvárí     : {result.num_unassigned}")
    print(f"Čas                     : {elapsed:.2f} s")

    if not result.clusters:
        return

    print()
    print(f"{'OSOBA':<12}{'TVÁRÍ':>7}{'FOTIEK':>8}   NÁHĽAD")
    print("-" * LINE_WIDTH)
    for cluster in result.person_clusters:
        preview = Path(cluster.representative_preview_path or "-").name
        print(f"{cluster.name:<12}{cluster.size:>7}{len(cluster.source_paths):>8}   {preview}")

    unassigned = result.unassigned_cluster
    if unassigned is not None:
        print(f"{unassigned.name:<12}{unassigned.size:>7}{len(unassigned.source_paths):>8}   -")

    header("DETAIL OSÔB")
    for cluster in result.person_clusters:
        print(f"\n{cluster.name} ({cluster.size} tvárí na {len(cluster.source_paths)} fotkách)")
        print(f"  reprezentant: {cluster.representative_preview_path}")
        for source in cluster.source_paths[:5]:
            count = sum(
                1
                for face_id in cluster.face_ids
                if clusterer_faces[face_id].source_path == source
            )
            suffix = f" (x{count})" if count > 1 else ""
            print(f"  - {Path(source).name}{suffix}")
        if len(cluster.source_paths) > 5:
            print(f"  ... a ďalších {len(cluster.source_paths) - 5} fotiek")


def main(argv: list[str] | None = None) -> int:
    configure_console()
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    folder: Path = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"CHYBA: '{folder}' nie je priečinok.", file=sys.stderr)
        return 2

    settings = get_settings()
    header("KONFIGURÁCIA")
    print(f"Priečinok      : {folder}")
    print(f"Model          : {settings.INSIGHTFACE_MODEL_NAME} (root={settings.INSIGHTFACE_ROOT})")
    print(f"Cache náhľadov : {settings.CACHE_DIR}")
    print(f"DBSCAN         : eps={args.eps or settings.DBSCAN_EPS}, "
          f"min_samples={args.min_samples or settings.DBSCAN_MIN_SAMPLES}, metric=cosine")

    images = list(FaceDetector.iter_images(str(folder), recursive=not args.no_recursive))
    if not images:
        print("\nV priečinku sa nenašli žiadne podporované obrázky.", file=sys.stderr)
        return 1
    print(f"Nájdených fotografií: {len(images)}")

    detector = FaceDetector(settings=settings)
    print("\nNačítavam model (prvý raz sa sťahuje ~300 MB)...")
    load_started = time.perf_counter()
    detector.warmup()
    print(f"Model pripravený za {time.perf_counter() - load_started:.2f} s")

    started = time.perf_counter()
    faces, failures = detector.process_directory(str(folder), recursive=not args.no_recursive)
    detection_elapsed = time.perf_counter() - started
    print_detection_stats(faces, failures, len(images), detection_elapsed)

    if not faces:
        print("\nŽiadne tváre - zhlukovanie preskočené.")
        return 0

    clusterer = FaceClusterer(
        settings=settings,
        eps=args.eps,
        min_samples=args.min_samples,
    )
    started = time.perf_counter()
    result = clusterer.cluster(faces)
    print_clustering_stats(result, faces, time.perf_counter() - started)

    header("HOTOVO")
    print(f"{len(faces)} tvárí z {len(images)} fotografií -> {result.num_persons} osôb "
          f"({result.num_unassigned} nepriradených)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
