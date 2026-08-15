"""Príkazový riadok appky.

    python -m app.cli scan    C:\\fotky            # detekcia (preskočí známe fotky)
    python -m app.cli cluster --eps 0.45          # prepočet osôb
    python -m app.cli sort    output --clean      # roztriedenie do priečinkov
    python -m app.cli run     C:\\fotky output    # všetko naraz
    python -m app.cli persons | rename | stats | reset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.console import LINE_WIDTH, configure_console, header, human_size
from app.db import Database
from app.pipeline import PhotoSorterPipeline
from app.schemas import ClusteringResult, OrganizeReport, ScanReport


# --------------------------------------------------------------------- výpisy

def _progress(index: int, total: int, path: Path, cached: bool) -> None:
    tag = "cache" if cached else "nové "
    print(f"  [{index:>4}/{total}] {tag}  {path.name}", flush=True)


def print_scan_report(report: ScanReport) -> None:
    header("SCAN")
    print(f"Fotiek v priečinku : {report.total_images}")
    print(f"Novo spracovaných  : {report.processed}  (+{report.new_faces} tvárí)")
    print(f"Preskočených (cache): {report.skipped_cached}")
    print(f"Chybných           : {len(report.failures)}")
    print(f"Čas                : {report.elapsed_seconds:.1f} s")
    for path, error in report.failures[:10]:
        print(f"  [chyba] {path}: {error}")
    if len(report.failures) > 10:
        print(f"  ... a ďalších {len(report.failures) - 10} chýb")


def print_clustering(result: ClusteringResult) -> None:
    header("ZHLUKOVANIE")
    print(f"Identifikovaných osôb : {result.num_persons}")
    print(f"Nepriradených tvárí   : {result.num_unassigned}")
    if result.person_clusters:
        print()
        print(f"{'OSOBA':<14}{'TVÁRÍ':>7}{'FOTIEK':>8}")
        print("-" * LINE_WIDTH)
        for cluster in result.person_clusters:
            print(f"{cluster.name:<14}{cluster.size:>7}{len(cluster.source_paths):>8}")


def print_organize_report(report: OrganizeReport) -> None:
    header("TRIEDENIE" + (" (DRY-RUN)" if report.dry_run else ""))
    print(f"Cieľ   : {report.output_dir}")
    print(f"Režim  : {report.mode}")
    print()
    print(f"{'PRIEČINOK':<28}{'FOTIEK':>8}")
    print("-" * LINE_WIDTH)
    for folder, count in report.folders.items():
        print(f"{folder:<28}{count:>8}")
    print("-" * LINE_WIDTH)
    print(f"{'SPOLU':<28}{report.total_photos:>8}")

    if not report.dry_run:
        print()
        print(f"Vytvorených položiek : {report.linked}")
        print(f"Už existovalo        : {report.skipped_existing}")
        if report.fallback_copies:
            print(f"Kópií namiesto linku : {report.fallback_copies}")
    for path, error in report.errors[:10]:
        print(f"  [chyba] {path}: {error}")


# -------------------------------------------------------------------- príkazy

def cmd_scan(args: argparse.Namespace) -> int:
    with PhotoSorterPipeline() as pipeline:
        report = pipeline.scan(
            args.folder,
            recursive=not args.no_recursive,
            force=args.force,
            progress=None if args.quiet else _progress,
        )
        print_scan_report(report)
    return 0 if not report.failures else 1


def cmd_cluster(args: argparse.Namespace) -> int:
    with PhotoSorterPipeline() as pipeline:
        result = pipeline.recluster(eps=args.eps, min_samples=args.min_samples)
        print_clustering(result)
    return 0


def cmd_sort(args: argparse.Namespace) -> int:
    with PhotoSorterPipeline() as pipeline:
        report = pipeline.organize(
            output_dir=args.output,
            mode=args.mode,
            dry_run=args.dry_run,
            clean=args.clean,
        )
        print_organize_report(report)
    return 0 if not report.errors else 1


def cmd_run(args: argparse.Namespace) -> int:
    with PhotoSorterPipeline() as pipeline:
        scan_report = pipeline.scan(
            args.folder,
            recursive=not args.no_recursive,
            force=args.force,
            progress=None if args.quiet else _progress,
        )
        print_scan_report(scan_report)

        result = pipeline.recluster(eps=args.eps, min_samples=args.min_samples)
        print_clustering(result)

        organize_report = pipeline.organize(
            output_dir=args.output,
            mode=args.mode,
            dry_run=args.dry_run,
            clean=args.clean,
        )
        print_organize_report(organize_report)

    header("HOTOVO")
    print(
        f"{scan_report.total_images} fotiek -> {result.num_persons} osôb "
        f"({result.num_unassigned} nepriradených tvárí)"
    )
    return 0


def cmd_persons(args: argparse.Namespace) -> int:
    with Database() as db:
        persons = db.list_persons()
        header("OSOBY")
        if not persons:
            print("Zatiaľ žiadne - spusti najprv 'scan' a 'cluster'.")
            return 0
        print(f"{'LABEL':<12}{'MENO':<20}{'TVÁRÍ':>7}   NÁHĽAD")
        print("-" * LINE_WIDTH)
        for person in persons:
            preview = Path(person.representative_face_id or "-").name
            print(
                f"{person.label:<12}{(person.display_name or '-'):<20}"
                f"{person.face_count:>7}   {preview}"
            )
    return 0


def cmd_rename(args: argparse.Namespace) -> int:
    with Database() as db:
        name = None if args.name.strip() in {"", "-"} else args.name.strip()
        if db.rename_person(args.label, name):
            print(f"{args.label} -> {name or '(bez mena)'}")
            return 0
        print(f"Osoba '{args.label}' neexistuje.", file=sys.stderr)
        return 1


def cmd_stats(args: argparse.Namespace) -> int:
    settings = get_settings()
    with Database() as db:
        stats = db.stats()
    header("DATABÁZA")
    print(f"Fotiek             : {stats.photos}  (bez tvárí: {stats.photos_without_faces})")
    print(f"Tvárí              : {stats.faces}")
    print(f"Priradených tvárí  : {stats.assigned_faces}  (nepriradených: {stats.unassigned_faces})")
    print(f"Osôb               : {stats.persons}")
    print(f"Súbor              : {stats.db_path}  ({human_size(stats.db_size_bytes)})")
    print(f"Náhľady tvárí      : {settings.CACHE_DIR}")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    if not args.yes:
        print("Zmaže obsah databázy. Potvrď prepínačom --yes.", file=sys.stderr)
        return 1
    with Database() as db:
        db.reset()
    print("Databáza vyprázdnená.")
    return 0


# --------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="photo-face-sorter",
        description="Automatické triedenie fotografií podľa tvárí.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Podrobné logovanie.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_scan_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("folder", type=Path, help="Priečinok s fotografiami.")
        sub.add_argument("--no-recursive", action="store_true", help="Bez podpriečinkov.")
        sub.add_argument("--force", action="store_true", help="Spracovať aj známe fotky.")
        sub.add_argument("--quiet", "-q", action="store_true", help="Bez priebežného výpisu.")

    def add_cluster_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--eps", type=float, default=None,
                         help=f"DBSCAN eps (default {settings.DBSCAN_EPS}).")
        sub.add_argument("--min-samples", type=int, default=None,
                         help=f"DBSCAN min_samples (default {settings.DBSCAN_MIN_SAMPLES}).")

    def add_sort_args(sub: argparse.ArgumentParser, positional: bool) -> None:
        if positional:
            sub.add_argument("output", type=Path, nargs="?", default=None,
                             help=f"Cieľový priečinok (default {settings.OUTPUT_DIR}).")
        else:
            sub.add_argument("--output", type=Path, default=None, help="Cieľový priečinok.")
        sub.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default=None,
                         help=f"Spôsob umiestnenia (default {settings.LINK_MODE}).")
        sub.add_argument("--dry-run", action="store_true", help="Len vypíše plán.")
        sub.add_argument("--clean", action="store_true", help="Zmaže predošlý výstup.")

    scan = subparsers.add_parser("scan", help="Detekcia tvárí (preskočí známe fotky).")
    add_scan_args(scan)
    scan.set_defaults(func=cmd_scan)

    cluster = subparsers.add_parser("cluster", help="Prepočíta osoby zo všetkých tvárí.")
    add_cluster_args(cluster)
    cluster.set_defaults(func=cmd_cluster)

    sort = subparsers.add_parser("sort", help="Roztriedi fotky do priečinkov podľa osôb.")
    add_sort_args(sort, positional=True)
    sort.set_defaults(func=cmd_sort)

    run = subparsers.add_parser("run", help="scan + cluster + sort v jednom kroku.")
    add_scan_args(run)
    add_cluster_args(run)
    run.add_argument("output", type=Path, nargs="?", default=None,
                     help=f"Cieľový priečinok (default {settings.OUTPUT_DIR}).")
    run.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default=None)
    run.add_argument("--dry-run", action="store_true", help="Netriedi, len vypíše plán.")
    run.add_argument("--clean", action="store_true", help="Zmaže predošlý výstup.")
    run.set_defaults(func=cmd_run)

    persons = subparsers.add_parser("persons", help="Vypíše nájdené osoby.")
    persons.set_defaults(func=cmd_persons)

    rename = subparsers.add_parser("rename", help="Pomenuje osobu (person_1 -> Mama).")
    rename.add_argument("label", help="Systémový názov osoby, napr. person_1.")
    rename.add_argument("name", help="Nové meno; '-' meno zruší.")
    rename.set_defaults(func=cmd_rename)

    stats = subparsers.add_parser("stats", help="Obsah databázy.")
    stats.set_defaults(func=cmd_stats)

    reset = subparsers.add_parser("reset", help="Vyprázdni databázu.")
    reset.add_argument("--yes", action="store_true", help="Potvrdenie.")
    reset.set_defaults(func=cmd_reset)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_console()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return int(args.func(args))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"CHYBA: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
