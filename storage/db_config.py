# storage/db_config.py
from __future__ import annotations
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

SUPPORTED_VECTOR_BACKENDS = ("none", "chroma", "faiss")

@dataclass
class DBConfig:
    base_dir: Path
    sqlite_path: Path
    vector_backend: str = "none"
    vector_settings: Dict[str, Any] = None

    @classmethod
    def from_env(cls) -> "DBConfig":
        base = Path(os.environ.get("STORAGE_DIR", "storage")).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        sqlite_path = base / os.environ.get("SQLITE_FILENAME", "storage_meta.sqlite")
        vector_backend = os.environ.get("VECTOR_BACKEND", "chroma").lower()
        vs_raw = os.environ.get("VECTOR_SETTINGS")
        vector_settings = {}
        if vs_raw:
            import json
            try:
                vector_settings = json.loads(vs_raw)
            except Exception:
                vector_settings = {}
        if vector_backend not in SUPPORTED_VECTOR_BACKENDS:
            raise ValueError(f"Unsupported VECTOR_BACKEND={vector_backend}")
        return cls(base_dir=base, sqlite_path=sqlite_path, vector_backend=vector_backend, vector_settings=vector_settings)

def create_sqlite_conn(db_path: Path, pragmas: Optional[Dict[str, Any]] = None) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    cur = conn.cursor()
    default_pragmas = {"journal_mode": "WAL", "synchronous": "NORMAL", "foreign_keys": 1}
    pragmas = {**default_pragmas, **(pragmas or {})}
    for k, v in pragmas.items():
        try:
            cur.execute(f"PRAGMA {k}={v}")
        except Exception:
            cur.execute(f"PRAGMA {k}='{v}'")
    conn.commit()
    return conn

def init_sqlite_schema(conn: sqlite3.Connection, create_embeddings_table: bool = False) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            file_hash TEXT,
            page_hash TEXT,
            page_number INTEGER,
            source_path TEXT,
            created_at INTEGER,
            metadata_json TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents (file_hash)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS status (
            file_hash TEXT PRIMARY KEY,
            state TEXT,
            details_json TEXT,
            updated_at INTEGER
        )
    """)
    conn.commit()

# -----------------------
# Chroma factory (safe)
# -----------------------
def get_chroma_client(cfg: DBConfig):
    """
    Return a chroma.Client configured to persist under cfg.base_dir/chroma (if chromadb is installed).
    If chromadb is not installed an ImportError is raised with actionable instructions.
    """
    try:
        import chromadb
    except Exception as exc:
        raise ImportError("Requested chroma backend but chromadb is not installed. pip install chromadb") from exc

    # try persistent storage dir under base_dir
    persist_dir = str(cfg.base_dir / "chroma")
    # inside storage/db_config.py, before calling chromadb.Client(...)
    try:
        import chromadb
        # defensive monkeypatch: harmless no-op to swallow unexpected telemetry calls
        try:
            chromadb.telemetry.capture = lambda *args, **kwargs: None
        except Exception:
            # some builds may not expose telemetry attribute; ignore
            pass
    except Exception:
        raise ImportError("Requested chroma backend but chromadb is not installed. pip install chromadb")

        
    try:
        # chromadb allows passing a persist_directory param on Client in common setups
        client = chromadb.Client(persist_directory=persist_dir)
    except TypeError:
        # older/newer API differences — fall back to no-arg client and instruct user to configure persist in settings
        client = chromadb.Client()
    return client
