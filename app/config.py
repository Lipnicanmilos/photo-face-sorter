"""Konfigurácia aplikácie načítaná z prostredia / .env súboru."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Nastavenia ML pipeline-u pre detekciu a zhlukovanie tvárí."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Zhlukovanie (DBSCAN) ---------------------------------------------
    DBSCAN_EPS: float = Field(
        default=0.5,
        gt=0.0,
        le=2.0,
        description=(
            "Maximálna kosínusová vzdialenosť medzi dvoma embeddingmi, aby boli "
            "považované za susedov. Nižšia hodnota = prísnejšie zhlukovanie."
        ),
    )
    DBSCAN_MIN_SAMPLES: int = Field(
        default=2,
        ge=1,
        description="Minimálny počet tvárí potrebný na vytvorenie zhluku (osoby).",
    )

    # --- Úložisko ----------------------------------------------------------
    CACHE_DIR: Path = Field(
        default=BASE_DIR / "cache" / "faces",
        description="Adresár pre orezané náhľady detegovaných tvárí.",
    )

    # --- InsightFace -------------------------------------------------------
    INSIGHTFACE_MODEL_NAME: str = Field(
        default="buffalo_l",
        description="Názov balíka modelov InsightFace (ArcFace, 512d embeddingy).",
    )
    INSIGHTFACE_ROOT: Path = Field(
        default=BASE_DIR,
        description=(
            "Koreň pre váhy InsightFace. Knižnica si doň sama vytvorí podpriečinok "
            "'models/{INSIGHTFACE_MODEL_NAME}' a pri prvom spustení tam stiahne ~300 MB "
            "modelov. Nastav na '~/.insightface', ak chceš zdieľanú systémovú cache."
        ),
    )
    INSIGHTFACE_CTX_ID: int = Field(
        default=-1,
        description="ID zariadenia: -1 = CPU, 0 a viac = index GPU.",
    )
    INSIGHTFACE_DET_SIZE: int = Field(
        default=640,
        ge=128,
        description="Rozlíšenie vstupu detektora (štvorec det_size x det_size).",
    )

    # --- Filtrovanie detekcií ---------------------------------------------
    MIN_DET_SCORE: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimálne skóre detekcie, pod ktorým sa tvár zahodí.",
    )
    MIN_FACE_SIZE: int = Field(
        default=32,
        ge=0,
        description="Minimálna šírka aj výška bounding boxu tváre v pixeloch.",
    )
    FACE_CROP_MARGIN: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Relatívny okraj pridaný okolo bounding boxu pri orezávaní náhľadu.",
    )
    PREVIEW_SIZE: int = Field(
        default=160,
        ge=32,
        description="Hrana štvorcového náhľadu tváre v pixeloch.",
    )
    PREVIEW_JPEG_QUALITY: int = Field(
        default=90,
        ge=1,
        le=100,
        description="Kvalita JPEG kompresie pre náhľady tvárí.",
    )

    @field_validator("CACHE_DIR", "INSIGHTFACE_ROOT", mode="after")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        """Rozvinie '~', relatívne cesty vyhodnotí voči koreňu projektu a normalizuje ich."""
        path = value.expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        return path.resolve()

    def ensure_directories(self) -> None:
        """Vytvorí potrebné adresáre, ak ešte neexistujú."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.INSIGHTFACE_ROOT.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Vráti singleton inštanciu nastavení (cachovanú počas behu procesu)."""
    return Settings()


settings: Settings = get_settings()
