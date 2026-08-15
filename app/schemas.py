"""Zdieľané dátové modely ML pipeline-u (Pydantic v2)."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

EMBEDDING_DIM: int = 512
"""Dimenzia ArcFace embeddingu produkovaného modelom buffalo_l."""

UNASSIGNED_LABEL: int = -1
"""DBSCAN označenie pre šum, t. j. tváre nepriradené k žiadnej osobe."""


class BoundingBox(BaseModel):
    """Súradnice tváre v pôvodnej fotografii (v pixeloch, ľavý horný roh = 0,0)."""

    model_config = ConfigDict(frozen=True)

    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


class DetectedFace(BaseModel):
    """Jedna detegovaná tvár aj s jej ArcFace embeddingom."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    face_id: str = Field(description="Unikátny identifikátor tváre (UUID hex).")
    source_path: str = Field(description="Cesta k pôvodnej fotografii.")
    bbox: BoundingBox = Field(description="Súradnice tváre v pôvodnej fotografii.")
    det_score: float = Field(ge=0.0, le=1.0, description="Istota detektora.")
    preview_path: str = Field(description="Cesta k orezanému náhľadu CACHE_DIR/{face_id}.jpg.")
    embedding: Annotated[np.ndarray, Field(description="L2-normalizovaný 512d vektor.")]
    age: int | None = Field(default=None, description="Odhadovaný vek (ak ho model poskytne).")
    gender: str | None = Field(default=None, description="Odhadované pohlavie: 'M' / 'F'.")

    def embedding_as_list(self) -> list[float]:
        """Embedding ako JSON-serializovateľný zoznam (napr. pre uloženie do DB)."""
        return [float(value) for value in self.embedding]


class FaceCluster(BaseModel):
    """Zhluk tvárí reprezentujúci jednu osobu (alebo skupinu nepriradených tvárí)."""

    label: int = Field(description="Interné DBSCAN označenie; -1 = nepriradené.")
    name: str = Field(description="Čitateľný názov, napr. 'person_1' alebo 'unassigned'.")
    face_ids: list[str] = Field(default_factory=list, description="Tváre patriace do zhluku.")
    representative_face_id: str | None = Field(
        default=None,
        description="Tvár najbližšie k centroidu zhluku - použije sa ako náhľad osoby.",
    )
    representative_preview_path: str | None = Field(
        default=None,
        description="Cesta k náhľadu reprezentatívnej tváre.",
    )
    source_paths: list[str] = Field(
        default_factory=list,
        description="Unikátne fotografie, na ktorých sa osoba vyskytuje.",
    )

    @property
    def size(self) -> int:
        return len(self.face_ids)

    @property
    def is_unassigned(self) -> bool:
        return self.label == UNASSIGNED_LABEL


class ClusteringResult(BaseModel):
    """Výsledok zhlukovania celej kolekcie tvárí."""

    clusters: list[FaceCluster] = Field(default_factory=list)
    labels_by_face_id: dict[str, int] = Field(
        default_factory=dict,
        description="Mapovanie face_id -> označenie zhluku (-1 = nepriradené).",
    )

    @property
    def person_clusters(self) -> list[FaceCluster]:
        """Zhluky reprezentujúce identifikované osoby (bez šumu)."""
        return [cluster for cluster in self.clusters if not cluster.is_unassigned]

    @property
    def unassigned_cluster(self) -> FaceCluster | None:
        """Zhluk nepriradených tvárí, ak nejaké existujú."""
        return next((cluster for cluster in self.clusters if cluster.is_unassigned), None)

    @property
    def num_persons(self) -> int:
        return len(self.person_clusters)

    @property
    def num_unassigned(self) -> int:
        cluster = self.unassigned_cluster
        return cluster.size if cluster is not None else 0


class PhotoRecord(BaseModel):
    """Riadok tabuľky `photos` - už spracovaná fotografia."""

    id: int
    path: str
    content_hash: str
    file_size: int
    num_faces: int
    processed_at: str


class PersonRecord(BaseModel):
    """Riadok tabuľky `persons` - osoba vzniknutá zhlukovaním."""

    id: int
    label: str = Field(description="Systémový názov zhluku, napr. 'person_1'.")
    display_name: str | None = Field(default=None, description="Ručne zadané meno osoby.")
    representative_face_id: str | None = None
    face_count: int = 0
    updated_at: str

    @property
    def folder_name(self) -> str:
        """Názov priečinka pri triedení - uprednostní ručne zadané meno."""
        return self.display_name or self.label


class StorageStats(BaseModel):
    """Prehľad obsahu databázy."""

    photos: int = 0
    photos_without_faces: int = 0
    faces: int = 0
    assigned_faces: int = 0
    persons: int = 0
    db_path: str = ""
    db_size_bytes: int = 0

    @property
    def unassigned_faces(self) -> int:
        return self.faces - self.assigned_faces


class ScanReport(BaseModel):
    """Výsledok prehľadania priečinka s fotografiami."""

    total_images: int = 0
    processed: int = 0
    skipped_cached: int = 0
    new_faces: int = 0
    failures: list[tuple[str, str]] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


class OrganizeReport(BaseModel):
    """Výsledok roztriedenia fotiek do priečinkov."""

    output_dir: str = ""
    mode: str = "hardlink"
    dry_run: bool = False
    folders: dict[str, int] = Field(
        default_factory=dict,
        description="Názov priečinka -> počet fotiek, ktoré doň patria.",
    )
    linked: int = Field(default=0, description="Počet vytvorených položiek vo výstupe.")
    skipped_existing: int = Field(default=0, description="Cieľ už existoval a sedel.")
    fallback_copies: int = Field(
        default=0,
        description="Koľkokrát sa hardlink/symlink nepodaril a použila sa kópia.",
    )
    errors: list[tuple[str, str]] = Field(default_factory=list)

    @property
    def total_photos(self) -> int:
        return sum(self.folders.values())
