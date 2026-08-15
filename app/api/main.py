"""FastAPI aplikácia - REST API nad pipeline-om plus statické webové UI.

Spustenie:
    python -m app.cli serve
    uvicorn app.api.main:app --reload
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.api.jobs import Job, JobBusyError, JobManager
from app.config import get_settings
from app.db import Database
from app.pipeline import PhotoSorterPipeline
from app.services.detector import FaceDetector
from app.services.organizer import OrganizeError

logger = logging.getLogger(__name__)

settings = get_settings()
settings.ensure_directories()

WEB_DIR: Path = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="Photo Face Sorter",
    description="Automatické triedenie fotografií podľa tvárí.",
    version="0.2.0",
)

jobs = JobManager()
_detector = FaceDetector(settings=settings)
"""Jeden zdieľaný detektor - model má ~300 MB, načíta sa raz a drží sa v pamäti."""


# ------------------------------------------------------------------ modely API

class ScanRequest(BaseModel):
    folder: str = Field(description="Priečinok s fotografiami.")
    recursive: bool = True
    force: bool = Field(default=False, description="Spracovať aj už známe fotky.")


class ClusterRequest(BaseModel):
    eps: float | None = Field(default=None, gt=0, le=2)
    min_samples: int | None = Field(default=None, ge=1)


class RenameRequest(BaseModel):
    display_name: str | None = Field(default=None, description="Prázdne meno ho zruší.")


class MergeRequest(BaseModel):
    source: str = Field(description="Osoba, ktorá zanikne.")
    target: str = Field(description="Osoba, do ktorej sa tváre presunú.")


class SortRequest(BaseModel):
    output: str | None = None
    mode: Literal["hardlink", "copy", "symlink"] | None = None
    dry_run: bool = False
    clean: bool = False


# -------------------------------------------------------------------- pomocné

def _db() -> Database:
    """Nové spojenie na požiadavku - SQLite spojenia sa nezdieľajú medzi vláknami."""
    return Database(settings=settings)


def _person_payload(db: Database) -> list[dict[str, Any]]:
    photo_counts = db.person_photo_counts()
    persons = db.list_persons()
    preview_by_face: dict[str, str] = {}
    for person in persons:
        if person.representative_face_id:
            preview_by_face[person.label] = f"{person.representative_face_id}.jpg"
    return [
        {
            "label": person.label,
            "display_name": person.display_name,
            "folder_name": person.folder_name,
            "face_count": person.face_count,
            "photo_count": photo_counts.get(person.id, 0),
            "preview_file": preview_by_face.get(person.label),
        }
        for person in persons
    ]


# ------------------------------------------------------------------- endpointy

@app.get("/api/stats")
def get_stats() -> dict[str, Any]:
    """Prehľad databázy a aktuálne nastavenia."""
    with _db() as db:
        stats = db.stats()
    active = jobs.active
    return {
        "photos": stats.photos,
        "photos_without_faces": stats.photos_without_faces,
        "faces": stats.faces,
        "assigned_faces": stats.assigned_faces,
        "unassigned_faces": stats.unassigned_faces,
        "persons": stats.persons,
        "db_path": stats.db_path,
        "db_size_bytes": stats.db_size_bytes,
        "defaults": {
            "eps": settings.DBSCAN_EPS,
            "min_samples": settings.DBSCAN_MIN_SAMPLES,
            "output_dir": str(settings.OUTPUT_DIR),
            "link_mode": settings.LINK_MODE,
        },
        "active_job": active.as_dict() if active else None,
    }


@app.post("/api/scan")
def start_scan(request: ScanRequest) -> dict[str, Any]:
    """Spustí detekciu tvárí na pozadí a vráti ID úlohy."""
    folder = Path(request.folder).expanduser()
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Priečinok neexistuje: {folder}")

    def work(job: Job) -> dict[str, Any]:
        with Database(settings=settings) as db:
            pipeline = PhotoSorterPipeline(db=db, settings=settings, detector=_detector)

            def progress(index: int, total: int, path: Path, cached: bool) -> None:
                job.current, job.total = index, total
                job.message = f"{'z cache' if cached else 'detegujem'}: {path.name}"

            job.message = "Načítavam model…"
            report = pipeline.scan(
                folder, recursive=request.recursive, force=request.force, progress=progress
            )
        return report.model_dump()

    try:
        job = jobs.start("scan", work)
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.as_dict()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """Stav úlohy na pozadí."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Úloha neexistuje.")
    return job.as_dict()


@app.post("/api/cluster")
def run_cluster(request: ClusterRequest) -> dict[str, Any]:
    """Prepočíta osoby zo všetkých tvárí v databáze."""
    with _db() as db:
        pipeline = PhotoSorterPipeline(db=db, settings=settings, detector=_detector)
        result = pipeline.recluster(eps=request.eps, min_samples=request.min_samples)
        return {
            "persons": result.num_persons,
            "unassigned": result.num_unassigned,
            "clusters": [
                {"name": cluster.name, "faces": cluster.size, "photos": len(cluster.source_paths)}
                for cluster in result.person_clusters
            ],
        }


@app.get("/api/persons")
def list_persons() -> dict[str, Any]:
    """Zoznam osôb s náhľadom a počtami."""
    with _db() as db:
        return {"persons": _person_payload(db), "unassigned": len(db.unassigned_faces())}


@app.get("/api/persons/{label}/faces")
def person_faces(label: str) -> dict[str, Any]:
    """Všetky tváre jednej osoby; `_unassigned` vráti nepriradené."""
    with _db() as db:
        if label == "_unassigned":
            faces = db.unassigned_faces()
        else:
            if db.get_person(label) is None:
                raise HTTPException(status_code=404, detail=f"Osoba '{label}' neexistuje.")
            faces = db.faces_of_person(label)
    return {"label": label, "faces": [face.model_dump() for face in faces]}


@app.patch("/api/persons/{label}")
def rename_person(label: str, request: RenameRequest) -> dict[str, Any]:
    """Premenuje osobu (prázdne meno vráti systémový názov)."""
    name = (request.display_name or "").strip() or None
    with _db() as db:
        if not db.rename_person(label, name):
            raise HTTPException(status_code=404, detail=f"Osoba '{label}' neexistuje.")
        return {"persons": _person_payload(db)}


@app.post("/api/persons/merge")
def merge_persons(request: MergeRequest) -> dict[str, Any]:
    """Zlúči dve osoby do jednej."""
    with _db() as db:
        if not db.merge_persons(request.source, request.target):
            raise HTTPException(
                status_code=400,
                detail="Zlúčenie sa nepodarilo - skontroluj, či obe osoby existujú a líšia sa.",
            )
        return {"persons": _person_payload(db)}


@app.post("/api/sort")
def sort_photos(request: SortRequest) -> dict[str, Any]:
    """Roztriedi fotky do priečinkov podľa osôb."""
    with _db() as db:
        pipeline = PhotoSorterPipeline(db=db, settings=settings, detector=_detector)
        try:
            report = pipeline.organize(
                output_dir=request.output or None,
                mode=request.mode,
                dry_run=request.dry_run,
                clean=request.clean,
            )
        except OrganizeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return report.model_dump()


@app.post("/api/reset")
def reset_database() -> dict[str, str]:
    """Vyprázdni databázu (náhľady tvárí na disku zostanú)."""
    with _db() as db:
        db.reset()
    return {"status": "ok"}


@app.get("/api/photo")
def get_photo(path: str = Query(description="Absolútna cesta k fotografii.")) -> FileResponse:
    """Pošle originálnu fotku - len ak je evidovaná v databáze.

    Kontrola voči databáze bráni tomu, aby sa cez tento endpoint dal prečítať
    ľubovoľný súbor na disku.
    """
    with _db() as db:
        known = {record.path for record in db.iter_photos()}
    if path not in known:
        raise HTTPException(status_code=404, detail="Fotka nie je v databáze.")

    file_path = Path(path)
    if not file_path.is_file():
        raise HTTPException(status_code=410, detail="Fotka už na disku nie je.")
    media_type, _ = mimetypes.guess_type(file_path.name)
    return FileResponse(file_path, media_type=media_type or "application/octet-stream")


# ------------------------------------------------------------ statické súbory

app.mount("/faces", StaticFiles(directory=str(settings.CACHE_DIR)), name="faces")


@app.get("/")
def index() -> FileResponse:
    """Webové UI."""
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.exception_handler(OrganizeError)
def _organize_error_handler(request: object, exc: OrganizeError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
