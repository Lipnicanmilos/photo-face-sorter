"""SQLite perzistencia: spracované fotky, tváre s embeddingmi a osoby.

Vďaka nej sa už raz spracovaná fotografia pri ďalšom behu preskočí - detekcia je
zďaleka najdrahšia časť pipeline-u (~10 s/fotku na CPU).
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

from app.config import Settings, get_settings
from app.schemas import (
    EMBEDDING_DIM,
    UNASSIGNED_LABEL,
    BoundingBox,
    ClusteringResult,
    DetectedFace,
    FaceSummary,
    PersonRecord,
    PhotoRecord,
    StorageStats,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1

_SCHEMA: str = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS photos (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT    NOT NULL UNIQUE,
    content_hash  TEXT    NOT NULL,
    file_size     INTEGER NOT NULL,
    num_faces     INTEGER NOT NULL DEFAULT 0,
    processed_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_photos_hash ON photos (content_hash);

CREATE TABLE IF NOT EXISTS persons (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    label                  TEXT    NOT NULL UNIQUE,
    display_name           TEXT,
    representative_face_id TEXT,
    face_count             INTEGER NOT NULL DEFAULT 0,
    updated_at             TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS faces (
    face_id      TEXT    PRIMARY KEY,
    photo_id     INTEGER NOT NULL REFERENCES photos (id)  ON DELETE CASCADE,
    person_id    INTEGER          REFERENCES persons (id) ON DELETE SET NULL,
    x1           INTEGER NOT NULL,
    y1           INTEGER NOT NULL,
    x2           INTEGER NOT NULL,
    y2           INTEGER NOT NULL,
    det_score    REAL    NOT NULL,
    preview_path TEXT    NOT NULL,
    embedding    BLOB    NOT NULL,
    age          INTEGER,
    gender       TEXT
);
CREATE INDEX IF NOT EXISTS idx_faces_photo  ON faces (photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces (person_id);
"""


def compute_file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 obsahu súboru - identifikuje fotku nezávisle od názvu a cesty."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Database:
    """Tenká vrstva nad SQLite - jedno spojenie, explicitné transakcie."""

    def __init__(self, settings: Settings | None = None, db_path: Path | None = None) -> None:
        cfg = settings or get_settings()
        self._settings: Settings = cfg
        self.path: Path = Path(db_path) if db_path is not None else cfg.DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript(_SCHEMA)
        self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._connection.commit()

    # ------------------------------------------------------------ životný cyklus

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commitne pri úspechu, rollbackne pri výnimke."""
        try:
            yield self._connection
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    # ------------------------------------------------------------------ fotky

    def is_photo_processed(self, path: Path, content_hash: str) -> bool:
        """True, ak je fotka v DB s rovnakým obsahom (t. j. netreba ju detegovať)."""
        row = self._connection.execute(
            "SELECT content_hash FROM photos WHERE path = ?", (str(path),)
        ).fetchone()
        return row is not None and row["content_hash"] == content_hash

    def save_photo_with_faces(
        self,
        path: Path,
        content_hash: str,
        file_size: int,
        faces: Sequence[DetectedFace],
    ) -> int:
        """Uloží fotku aj jej tváre; existujúci záznam nahradí (re-scan po zmene).

        Returns:
            ID fotky v tabuľke `photos`.
        """
        with self.transaction() as conn:
            conn.execute("DELETE FROM photos WHERE path = ?", (str(path),))
            cursor = conn.execute(
                "INSERT INTO photos (path, content_hash, file_size, num_faces, processed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(path), content_hash, file_size, len(faces), _now()),
            )
            photo_id = int(cursor.lastrowid)

            conn.executemany(
                "INSERT INTO faces (face_id, photo_id, person_id, x1, y1, x2, y2, "
                "det_score, preview_path, embedding, age, gender) "
                "VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        face.face_id,
                        photo_id,
                        face.bbox.x1,
                        face.bbox.y1,
                        face.bbox.x2,
                        face.bbox.y2,
                        face.det_score,
                        face.preview_path,
                        np.asarray(face.embedding, dtype=np.float32).tobytes(),
                        face.age,
                        face.gender,
                    )
                    for face in faces
                ],
            )
        return photo_id

    def iter_photos(self) -> Iterator[PhotoRecord]:
        """Všetky spracované fotky zoradené podľa cesty."""
        for row in self._connection.execute(
            "SELECT id, path, content_hash, file_size, num_faces, processed_at "
            "FROM photos ORDER BY path"
        ):
            yield PhotoRecord(**dict(row))

    def photo_paths_without_faces(self) -> list[str]:
        """Fotky, na ktorých detektor nenašiel ani jednu tvár."""
        return [
            row["path"]
            for row in self._connection.execute(
                "SELECT path FROM photos WHERE num_faces = 0 ORDER BY path"
            )
        ]

    def remove_missing_photos(self) -> int:
        """Zmaže z DB fotky, ktoré už na disku nie sú. Vracia počet zmazaných."""
        stale = [
            row["id"]
            for row in self._connection.execute("SELECT id, path FROM photos")
            if not Path(row["path"]).exists()
        ]
        if stale:
            with self.transaction() as conn:
                conn.executemany("DELETE FROM photos WHERE id = ?", [(pid,) for pid in stale])
        return len(stale)

    # ------------------------------------------------------------------ tváre

    def load_faces(self) -> list[DetectedFace]:
        """Načíta všetky tváre aj s embeddingmi (vstup pre zhlukovanie)."""
        rows = self._connection.execute(
            "SELECT f.face_id, f.x1, f.y1, f.x2, f.y2, f.det_score, f.preview_path, "
            "       f.embedding, f.age, f.gender, p.path AS source_path "
            "FROM faces f JOIN photos p ON p.id = f.photo_id "
            "ORDER BY p.path, f.face_id"
        ).fetchall()
        return [self._row_to_face(row) for row in rows]

    @staticmethod
    def _row_to_face(row: sqlite3.Row) -> DetectedFace:
        embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        if embedding.size != EMBEDDING_DIM:
            raise ValueError(
                f"Poškodený embedding pre tvár {row['face_id']}: {embedding.size} hodnôt."
            )
        return DetectedFace(
            face_id=row["face_id"],
            source_path=row["source_path"],
            bbox=BoundingBox(x1=row["x1"], y1=row["y1"], x2=row["x2"], y2=row["y2"]),
            det_score=row["det_score"],
            preview_path=row["preview_path"],
            embedding=embedding.copy(),  # frombuffer je read-only
            age=row["age"],
            gender=row["gender"],
        )

    # ------------------------------------------------------------------ osoby

    def save_clustering(self, result: ClusteringResult) -> None:
        """Prepíše osoby výsledkom nového zhlukovania.

        Ručne zadané mená (`display_name`) sa prenesú na nový zhluk, ktorý zdedil
        najviac tvárí od pôvodnej osoby - inak by ich prečíslovanie zhlukov stratilo.
        """
        inherited = self._inherited_display_names(result)

        with self.transaction() as conn:
            conn.execute("UPDATE faces SET person_id = NULL")
            conn.execute("DELETE FROM persons")

            for cluster in result.clusters:
                if cluster.is_unassigned:
                    continue
                cursor = conn.execute(
                    "INSERT INTO persons (label, display_name, representative_face_id, "
                    "face_count, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        cluster.name,
                        inherited.get(cluster.name),
                        cluster.representative_face_id,
                        cluster.size,
                        _now(),
                    ),
                )
                person_id = int(cursor.lastrowid)
                conn.executemany(
                    "UPDATE faces SET person_id = ? WHERE face_id = ?",
                    [(person_id, face_id) for face_id in cluster.face_ids],
                )

    def _inherited_display_names(self, result: ClusteringResult) -> dict[str, str]:
        """Namapuje existujúce `display_name` na nové zhluky podľa prekryvu tvárí."""
        rows = self._connection.execute(
            "SELECT f.face_id, p.display_name FROM faces f "
            "JOIN persons p ON p.id = f.person_id WHERE p.display_name IS NOT NULL"
        ).fetchall()
        if not rows:
            return {}

        name_by_face: dict[str, str] = {row["face_id"]: row["display_name"] for row in rows}
        inherited: dict[str, str] = {}
        used: set[str] = set()

        for cluster in sorted(result.person_clusters, key=lambda c: c.size, reverse=True):
            votes: dict[str, int] = {}
            for face_id in cluster.face_ids:
                name = name_by_face.get(face_id)
                if name is not None and name not in used:
                    votes[name] = votes.get(name, 0) + 1
            if votes:
                winner = max(votes.items(), key=lambda item: item[1])[0]
                inherited[cluster.name] = winner
                used.add(winner)
        return inherited

    def list_persons(self) -> list[PersonRecord]:
        """Osoby zoradené od najväčšej po najmenšiu."""
        rows = self._connection.execute(
            "SELECT id, label, display_name, representative_face_id, face_count, updated_at "
            "FROM persons ORDER BY face_count DESC, id"
        ).fetchall()
        return [PersonRecord(**dict(row)) for row in rows]

    def rename_person(self, label: str, display_name: str | None) -> bool:
        """Nastaví (alebo zruší) ľudské meno osoby. False, ak `label` neexistuje."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE persons SET display_name = ?, updated_at = ? WHERE label = ?",
                (display_name, _now(), label),
            )
        return cursor.rowcount > 0

    def get_person(self, label: str) -> PersonRecord | None:
        """Nájde osobu podľa systémového názvu (`person_1`)."""
        row = self._connection.execute(
            "SELECT id, label, display_name, representative_face_id, face_count, updated_at "
            "FROM persons WHERE label = ?",
            (label,),
        ).fetchone()
        return PersonRecord(**dict(row)) if row is not None else None

    def person_photo_counts(self) -> dict[int, int]:
        """Mapovanie `person.id` -> počet unikátnych fotiek s touto osobou."""
        return {
            int(row["person_id"]): int(row["photos"])
            for row in self._connection.execute(
                "SELECT person_id, COUNT(DISTINCT photo_id) AS photos FROM faces "
                "WHERE person_id IS NOT NULL GROUP BY person_id"
            )
        }

    def faces_of_person(self, label: str) -> list[FaceSummary]:
        """Tváre patriace osobe, zoradené od najistejšej detekcie."""
        rows = self._connection.execute(
            "SELECT f.face_id, f.preview_path, f.det_score, p.path AS source_path "
            "FROM faces f JOIN photos p ON p.id = f.photo_id "
            "JOIN persons pe ON pe.id = f.person_id "
            "WHERE pe.label = ? ORDER BY f.det_score DESC",
            (label,),
        ).fetchall()
        return [self._row_to_summary(row, label) for row in rows]

    def unassigned_faces(self) -> list[FaceSummary]:
        """Tváre, ktoré zhlukovanie nepriradilo k žiadnej osobe."""
        rows = self._connection.execute(
            "SELECT f.face_id, f.preview_path, f.det_score, p.path AS source_path "
            "FROM faces f JOIN photos p ON p.id = f.photo_id "
            "WHERE f.person_id IS NULL ORDER BY f.det_score DESC"
        ).fetchall()
        return [self._row_to_summary(row, None) for row in rows]

    @staticmethod
    def _row_to_summary(row: sqlite3.Row, label: str | None) -> FaceSummary:
        return FaceSummary(
            face_id=row["face_id"],
            preview_file=Path(row["preview_path"]).name,
            source_path=row["source_path"],
            det_score=row["det_score"],
            person_label=label,
        )

    def merge_persons(self, source_label: str, target_label: str) -> bool:
        """Presunie všetky tváre zo `source` do `target` a zdrojovú osobu zmaže.

        Pozor: ďalšie spustenie zhlukovania počíta osoby odznova, takže ručné
        zlúčenie prežije len meno cieľovej osoby, nie samotné spojenie zhlukov.
        """
        source = self.get_person(source_label)
        target = self.get_person(target_label)
        if source is None or target is None or source.id == target.id:
            return False

        with self.transaction() as conn:
            conn.execute(
                "UPDATE faces SET person_id = ? WHERE person_id = ?", (target.id, source.id)
            )
            conn.execute("DELETE FROM persons WHERE id = ?", (source.id,))
            conn.execute(
                "UPDATE persons SET face_count = ("
                "  SELECT COUNT(*) FROM faces WHERE person_id = ?"
                "), updated_at = ? WHERE id = ?",
                (target.id, _now(), target.id),
            )
        logger.info("Osoba %s zlúčená do %s.", source_label, target_label)
        return True

    def photo_paths_by_person(self) -> dict[int, list[str]]:
        """Mapovanie `person.id` -> unikátne cesty fotiek, na ktorých osoba je."""
        result: dict[int, list[str]] = {}
        for row in self._connection.execute(
            "SELECT DISTINCT f.person_id AS person_id, p.path AS path "
            "FROM faces f JOIN photos p ON p.id = f.photo_id "
            "WHERE f.person_id IS NOT NULL ORDER BY p.path"
        ):
            result.setdefault(int(row["person_id"]), []).append(row["path"])
        return result

    def photo_paths_unassigned(self) -> list[str]:
        """Fotky, ktoré majú tváre, ale ani jedna nepatrí k žiadnej osobe."""
        return [
            row["path"]
            for row in self._connection.execute(
                "SELECT p.path AS path FROM photos p "
                "WHERE p.num_faces > 0 AND NOT EXISTS ("
                "  SELECT 1 FROM faces f WHERE f.photo_id = p.id AND f.person_id IS NOT NULL"
                ") ORDER BY p.path"
            )
        ]

    # ---------------------------------------------------------------- štatistiky

    def stats(self) -> StorageStats:
        """Prehľad obsahu databázy."""
        scalar = lambda sql: int(self._connection.execute(sql).fetchone()[0])  # noqa: E731
        return StorageStats(
            photos=scalar("SELECT COUNT(*) FROM photos"),
            photos_without_faces=scalar("SELECT COUNT(*) FROM photos WHERE num_faces = 0"),
            faces=scalar("SELECT COUNT(*) FROM faces"),
            assigned_faces=scalar("SELECT COUNT(*) FROM faces WHERE person_id IS NOT NULL"),
            persons=scalar("SELECT COUNT(*) FROM persons"),
            db_path=str(self.path),
            db_size_bytes=self.path.stat().st_size if self.path.exists() else 0,
        )

    def reset(self) -> None:
        """Vyprázdni databázu (fotky, tváre aj osoby)."""
        with self.transaction() as conn:
            conn.execute("DELETE FROM faces")
            conn.execute("DELETE FROM persons")
            conn.execute("DELETE FROM photos")
        self._connection.execute("VACUUM")
        logger.info("Databáza %s bola vyprázdnená.", self.path)


__all__ = ["Database", "compute_file_hash", "UNASSIGNED_LABEL", "SCHEMA_VERSION"]
