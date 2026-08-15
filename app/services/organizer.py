"""Roztriedenie fotografií do priečinkov podľa rozpoznaných osôb."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Literal

from app.config import Settings, get_settings
from app.db import Database
from app.schemas import OrganizeReport

logger = logging.getLogger(__name__)

LinkMode = Literal["hardlink", "copy", "symlink"]

MANIFEST_NAME: str = ".photo-face-sorter.json"
"""Zoznam priečinkov, ktoré appka vo výstupe vytvorila - aby ich vedela bezpečne prečistiť."""

_INVALID_CHARS: str = '<>:"/\\|?*'


class OrganizeError(RuntimeError):
    """Triedenie sa nedá bezpečne vykonať."""


class PhotoOrganizer:
    """Rozloží fotky do `OUTPUT_DIR/{osoba}/` podľa priradení uložených v databáze.

    Fotka s viacerými osobami sa objaví v priečinku každej z nich. Preto je
    predvolený režim `hardlink` - fotka existuje na disku raz, bez ohľadu na to,
    v koľkých priečinkoch sa zobrazuje.
    """

    def __init__(self, db: Database, settings: Settings | None = None) -> None:
        self._db: Database = db
        self._settings: Settings = settings or get_settings()

    # -------------------------------------------------------------- verejné API

    def plan(self) -> dict[str, list[str]]:
        """Vráti mapovanie `názov priečinka -> cesty fotiek` bez zápisu na disk."""
        cfg = self._settings
        paths_by_person = self._db.photo_paths_by_person()

        plan: dict[str, list[str]] = {}
        used_names: set[str] = set()
        for person in self._db.list_persons():
            photos = paths_by_person.get(person.id, [])
            if not photos:
                continue
            folder = self._unique_folder_name(person.folder_name, person.label, used_names)
            plan[folder] = photos

        unassigned = self._db.photo_paths_unassigned()
        if unassigned:
            plan[cfg.UNASSIGNED_DIR_NAME] = unassigned

        no_faces = self._db.photo_paths_without_faces()
        if no_faces:
            plan[cfg.NO_FACES_DIR_NAME] = no_faces

        return plan

    def organize(
        self,
        output_dir: Path | None = None,
        mode: LinkMode | None = None,
        dry_run: bool = False,
        clean: bool = False,
    ) -> OrganizeReport:
        """Roztriedi fotky do priečinkov podľa osôb.

        Args:
            output_dir: Cieľový adresár (default `OUTPUT_DIR` z konfigurácie).
            mode: 'hardlink', 'copy' alebo 'symlink'.
            dry_run: Nič nezapisuje, len vráti plán.
            clean: Pred triedením zmaže priečinky, ktoré appka vytvorila minule.

        Raises:
            OrganizeError: Ak je cieľový adresár obsadený cudzím obsahom.
        """
        cfg = self._settings
        target_root = Path(output_dir) if output_dir is not None else cfg.OUTPUT_DIR
        link_mode: LinkMode = mode or cfg.LINK_MODE

        plan = self.plan()
        report = OrganizeReport(
            output_dir=str(target_root),
            mode=link_mode,
            dry_run=dry_run,
            folders={folder: len(paths) for folder, paths in plan.items()},
        )
        if not plan:
            logger.warning("Niet čo triediť - v databáze nie sú žiadne fotky.")
            return report
        if dry_run:
            return report

        self._prepare_root(target_root, clean=clean)

        for folder, photo_paths in plan.items():
            folder_path = target_root / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            taken: set[str] = {item.name for item in folder_path.iterdir()}

            for photo_path in photo_paths:
                source = Path(photo_path)
                if not source.exists():
                    report.errors.append((photo_path, "zdrojová fotka už neexistuje"))
                    continue

                target = folder_path / self._unique_file_name(source.name, taken)
                try:
                    outcome = self._place(source, target, link_mode)
                except OSError as exc:
                    report.errors.append((photo_path, f"{type(exc).__name__}: {exc}"))
                    continue

                taken.add(target.name)
                if outcome == "skipped":
                    report.skipped_existing += 1
                else:
                    report.linked += 1
                    if outcome == "fallback":
                        report.fallback_copies += 1

        self._write_manifest(target_root, sorted(plan))
        logger.info(
            "Roztriedené do %s: %d priečinkov, %d položiek (%d preskočených, %d kópií navyše)",
            target_root,
            len(plan),
            report.linked,
            report.skipped_existing,
            report.fallback_copies,
        )
        return report

    # ---------------------------------------------------------------- pomocné

    def _prepare_root(self, root: Path, clean: bool) -> None:
        """Vytvorí cieľový adresár a voliteľne z neho odstráni predošlý výstup."""
        manifest_path = root / MANIFEST_NAME

        if root.exists() and not root.is_dir():
            raise OrganizeError(f"Cieľ '{root}' nie je priečinok.")

        if root.exists() and any(root.iterdir()) and not manifest_path.exists():
            raise OrganizeError(
                f"Priečinok '{root}' už obsahuje dáta, ktoré nevytvoril tento nástroj. "
                "Zvoľ prázdny alebo iný cieľ, aby sa nič neprepísalo."
            )

        if clean and manifest_path.exists():
            for folder in self._read_manifest(manifest_path):
                stale = root / folder
                if stale.is_dir():
                    shutil.rmtree(stale)
            logger.info("Predošlý výstup v %s bol odstránený.", root)

        root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read_manifest(manifest_path: Path) -> list[str]:
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            folders = data.get("folders", [])
            return [str(name) for name in folders if isinstance(name, str)]
        except (OSError, ValueError, AttributeError):
            logger.warning("Manifest %s sa nedá prečítať - preskakujem čistenie.", manifest_path)
            return []

    @staticmethod
    def _write_manifest(root: Path, folders: list[str]) -> None:
        payload = {"tool": "photo-face-sorter", "folders": folders}
        (root / MANIFEST_NAME).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _place(source: Path, target: Path, mode: LinkMode) -> str:
        """Umiestni jednu fotku. Vracia 'linked', 'skipped' alebo 'fallback'."""
        if target.exists():
            try:
                if target.samefile(source):
                    return "skipped"
            except OSError:
                pass
            if target.stat().st_size == source.stat().st_size:
                return "skipped"

        if mode == "copy":
            shutil.copy2(source, target)
            return "linked"

        try:
            if mode == "hardlink":
                os.link(source, target)
            else:
                target.symlink_to(source)
            return "linked"
        except (OSError, NotImplementedError) as exc:
            # Iný disk, chýbajúce oprávnenia (symlink na Windows) alebo FS bez podpory.
            logger.debug("%s zlyhal (%s), kopírujem: %s", mode, exc, source.name)
            shutil.copy2(source, target)
            return "fallback"

    @staticmethod
    def _sanitize(name: str) -> str:
        """Očistí meno osoby na názov priečinka platný aj na Windows."""
        cleaned = "".join("_" if char in _INVALID_CHARS else char for char in name)
        cleaned = cleaned.strip().rstrip(".")
        return cleaned or "osoba"

    def _unique_folder_name(self, preferred: str, fallback: str, used: set[str]) -> str:
        """Zaistí, že dve osoby nezdieľajú priečinok (napr. po rovnakom premenovaní)."""
        name = self._sanitize(preferred)
        if name in used:
            name = self._sanitize(f"{name}_{fallback}")
        counter = 2
        base = name
        while name in used:
            name = f"{base}_{counter}"
            counter += 1
        used.add(name)
        return name

    @staticmethod
    def _unique_file_name(file_name: str, taken: set[str]) -> str:
        """Rieši zhodu názvov fotiek z rôznych zdrojových priečinkov."""
        if file_name not in taken:
            return file_name
        stem, suffix = Path(file_name).stem, Path(file_name).suffix
        counter = 1
        while f"{stem}_{counter}{suffix}" in taken:
            counter += 1
        return f"{stem}_{counter}{suffix}"
