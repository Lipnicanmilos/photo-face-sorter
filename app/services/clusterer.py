"""Zhlukovanie tvárí do osôb pomocou DBSCAN nad ArcFace embeddingmi."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
from sklearn.cluster import DBSCAN

from app.config import Settings, get_settings
from app.schemas import (
    EMBEDDING_DIM,
    UNASSIGNED_LABEL,
    ClusteringResult,
    DetectedFace,
    FaceCluster,
)

logger = logging.getLogger(__name__)

UNASSIGNED_NAME: str = "unassigned"


class FaceClusterer:
    """Zoskupí embeddingy tvárí do osôb pomocou hustotného zhlukovania DBSCAN.

    Embeddingy sa porovnávajú kosínusovou vzdialenosťou (`1 - cos_sim`), takže
    `eps` sa dá čítať priamo ako "maximálna vzdialenosť dvoch tvárí tej istej osoby".
    """

    def __init__(
        self,
        settings: Settings | None = None,
        eps: float | None = None,
        min_samples: int | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self._settings: Settings = cfg
        self.eps: float = float(eps if eps is not None else cfg.DBSCAN_EPS)
        self.min_samples: int = int(
            min_samples if min_samples is not None else cfg.DBSCAN_MIN_SAMPLES
        )

    # -------------------------------------------------------------- verejné API

    def cluster(self, faces: Sequence[DetectedFace]) -> ClusteringResult:
        """Zhlukne detegované tváre a priradí ich k osobám.

        Args:
            faces: Detegované tváre s 512d embeddingmi.

        Returns:
            `ClusteringResult` so zoznamom zhlukov (`person_1`, `person_2`, ...,
            plus prípadný `unassigned`) a mapovaním `face_id -> label`.
        """
        if not faces:
            logger.info("Žiadne tváre na zhlukovanie.")
            return ClusteringResult()

        embeddings = self._stack_embeddings(faces)
        raw_labels = self.cluster_embeddings(embeddings)

        clusters = self._build_clusters(faces, embeddings, raw_labels)
        labels_by_face_id = {
            face_id: cluster.label for cluster in clusters for face_id in cluster.face_ids
        }

        logger.info(
            "Zhlukovanie hotové: %d tvárí -> %d osôb, %d nepriradených (eps=%.3f, min_samples=%d)",
            len(faces),
            sum(1 for cluster in clusters if not cluster.is_unassigned),
            sum(cluster.size for cluster in clusters if cluster.is_unassigned),
            self.eps,
            self.min_samples,
        )
        return ClusteringResult(clusters=clusters, labels_by_face_id=labels_by_face_id)

    def cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Spustí DBSCAN nad maticou embeddingov a vráti surové DBSCAN označenia.

        Args:
            embeddings: Pole tvaru `(n_faces, 512)` s L2-normalizovanými vektormi.

        Returns:
            Pole tvaru `(n_faces,)` s označeniami; `-1` znamená šum (nepriradené).
        """
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Očakávam maticu tvaru (n, {EMBEDDING_DIM}), dostal som {matrix.shape}."
            )
        if matrix.shape[0] == 0:
            return np.empty((0,), dtype=int)

        model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="cosine",
            n_jobs=-1,
        )
        return model.fit_predict(matrix).astype(int)

    # ---------------------------------------------------------------- pomocné

    @staticmethod
    def _stack_embeddings(faces: Sequence[DetectedFace]) -> np.ndarray:
        """Poskladá embeddingy do matice a znovu ich L2-normalizuje."""
        matrix = np.vstack([np.asarray(face.embedding, dtype=np.float32) for face in faces])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    def _build_clusters(
        self,
        faces: Sequence[DetectedFace],
        embeddings: np.ndarray,
        raw_labels: np.ndarray,
    ) -> list[FaceCluster]:
        """Premení DBSCAN označenia na pomenované zhluky zoradené podľa veľkosti."""
        indices_by_label: dict[int, list[int]] = OrderedDict()
        for index, label in enumerate(raw_labels.tolist()):
            indices_by_label.setdefault(int(label), []).append(index)

        person_labels = sorted(
            (label for label in indices_by_label if label != UNASSIGNED_LABEL),
            key=lambda label: (-len(indices_by_label[label]), label),
        )

        clusters: list[FaceCluster] = []
        for position, label in enumerate(person_labels, start=1):
            indices = indices_by_label[label]
            clusters.append(
                self._make_cluster(
                    label=position,  # prečíslované na stabilné 1..N
                    name=f"person_{position}",
                    faces=faces,
                    embeddings=embeddings,
                    indices=indices,
                    use_centroid=True,
                )
            )

        if UNASSIGNED_LABEL in indices_by_label:
            clusters.append(
                self._make_cluster(
                    label=UNASSIGNED_LABEL,
                    name=UNASSIGNED_NAME,
                    faces=faces,
                    embeddings=embeddings,
                    indices=indices_by_label[UNASSIGNED_LABEL],
                    use_centroid=False,
                )
            )

        return clusters

    def _make_cluster(
        self,
        label: int,
        name: str,
        faces: Sequence[DetectedFace],
        embeddings: np.ndarray,
        indices: list[int],
        use_centroid: bool,
    ) -> FaceCluster:
        """Zostaví jeden zhluk vrátane výberu reprezentatívneho náhľadu."""
        members = [faces[index] for index in indices]
        representative = (
            self._pick_representative(embeddings, indices, members)
            if use_centroid
            else max(members, key=lambda face: face.det_score)
        )

        source_paths = list(dict.fromkeys(face.source_path for face in members))
        return FaceCluster(
            label=label,
            name=name,
            face_ids=[face.face_id for face in members],
            representative_face_id=representative.face_id,
            representative_preview_path=representative.preview_path,
            source_paths=source_paths,
        )

    @staticmethod
    def _pick_representative(
        embeddings: np.ndarray,
        indices: list[int],
        members: Sequence[DetectedFace],
    ) -> DetectedFace:
        """Vyberie tvár najbližšie k centroidu zhluku (pri zhode rozhodne det_score).

        Takáto tvár je "najtypickejšia" pre danú osobu, čiže lepší náhľad než
        napr. prvá alebo najväčšia tvár v zhluku.
        """
        cluster_embeddings = embeddings[indices]
        centroid = cluster_embeddings.mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm == 0.0:
            return max(members, key=lambda face: face.det_score)

        similarities = cluster_embeddings @ (centroid / centroid_norm)
        best_position = max(
            range(len(members)),
            key=lambda position: (float(similarities[position]), members[position].det_score),
        )
        return members[best_position]
