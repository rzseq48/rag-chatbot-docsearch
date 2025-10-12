"""
data_ingestion/loader.py

Purpose
-------
High-level loader orchestrator for rag-chatbot-docsearch.

Responsibilities (summary)
- Discover candidate files (paths, globs, upload file-likes).
- Produce canonical FileDescriptor dictionaries for discovered inputs.
- Choose appropriate Handler for each file type.
- Stream page-level raw extractions (text or images) as Document records for downstream processing.
- Provide utilities for temp-writing uploads, quick hashing, mime sniffing, and utf heuristics.

Notes
-----
This file intentionally mixes:
- small, safe helper implementations you can reuse immediately, and
- well-commented skeleton classes/methods you will implement step-by-step.

Design guidance:
- Keep extraction (loader/handlers) separate from cleaning/chunking.
- Cache expensive steps (rendering, OCR results) in your cache_dir.
- Persist per-file status/manifest in a small sqlite/jsonl store (StatusStore below is a placeholder).
"""

from pathlib import Path
from typing import Iterator, Dict, Optional, Any, Iterable, Union, List, Tuple
import tempfile
import hashlib
import mimetypes
import logging
import fnmatch
import os
import io
import time
import abc
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Optional enhancement: if python-magic (libmagic) is installed we use it for better sniffing.
try:
    import magic  # type: ignore
    HAS_MAGIC = True
except Exception:
    HAS_MAGIC = False

# ---------------------------
# Configurable defaults
# ---------------------------
QUICK_HASH_WINDOW = 64 * 1024  # bytes read for quick hashing (64KB)
MIME_SNIFF_BYTES = 512
RECURSIVE_BY_DEFAULT = True
IGNORE_PATTERNS_DEFAULT: List[str] = ["*.DS_Store", "*.tmp", "*.~*"]

# extension groups
TEXT_EXTS = {".txt", ".md", ".csv", ".log"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx", ".pptx", ".xlsx"}
ARCHIVE_EXTS = {".zip", ".tar", ".tar.gz", ".tgz"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}

# ---------------------------
# Helper utilities (implemented)
# ---------------------------


def _is_file_like(obj: Any) -> bool:
    """
    Heuristic to decide whether obj is a readable file-like object.
    We prefer objects exposing read() (and optionally seek()).
    """
    return hasattr(obj, "read") and callable(getattr(obj, "read"))


def _read_head_bytes_from_path(path: Path, n: int) -> bytes:
    """Return first n bytes of a file at path. Return b'' on error."""
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except Exception:
        return b""


def _read_head_bytes_from_filelike(fobj: io.IOBase, n: int) -> bytes:
    """
    Read first n bytes from a file-like. If seekable, restore position afterwards.
    This is safe for UploadFile / io.BytesIO objects.
    """
    try:
        pos = fobj.tell()
    except Exception:
        pos = None

    try:
        data = fobj.read(n) or b""
    except Exception:
        data = b""

    try:
        if pos is not None:
            fobj.seek(pos)
    except Exception:
        pass
    # If data is str, encode
    if isinstance(data, str):
        return data.encode("utf-8", errors="replace")
    return data


def _sniff_mime(bytes_head: bytes, path: Optional[Path] = None) -> str:
    """
    Best-effort mime-type sniffing.
    - If python-magic available, use it on bytes_head.
    - Otherwise fall back to mimetypes.guess_type(path).
    Returns empty string if unknown.
    """
    if HAS_MAGIC and bytes_head:
        try:
            m = magic.Magic(mime=True)
            return m.from_buffer(bytes_head) or ""
        except Exception:
            pass
    if path is not None:
        guess, _ = mimetypes.guess_type(str(path))
        return guess or ""
    return ""


def _quick_hash_of(path: Optional[Path], filelike: Optional[io.IOBase], window: int = QUICK_HASH_WINDOW) -> str:
    """
    Compute a fast 'quick hash' to detect changes or dedupe quickly.
    Strategy:
    - If path provided: read first `window` bytes, plus append file size and mtime for more entropy.
    - If filelike provided: read first `window` bytes from the stream (attempt to preserve position).
    Returns hex sha256 digest.
    """
    h = hashlib.sha256()
    if path is not None:
        try:
            with open(path, "rb") as f:
                head = f.read(window)
                h.update(head)
        except Exception:
            # still return a valid digest
            h.update(b"")
        try:
            stat = path.stat()
            h.update(str(stat.st_size).encode())
            h.update(str(int(stat.st_mtime)).encode())
        except Exception:
            pass
    elif filelike is not None:
        head = _read_head_bytes_from_filelike(filelike, window)
        h.update(head)
    else:
        h.update(b"")
    return h.hexdigest()


def _utf_likely_from_bytes(head: bytes) -> bool:
    """
    Heuristic whether bytes appear to be text (utf-8 friendly).
    - If many NUL bytes -> likely binary
    - Try decoding as utf-8; if success, True. If latin-1 decodes but utf fails, return False (text but not utf).
    """
    if not head:
        return True
    nul_count = head.count(b"\x00")
    if len(head) and (nul_count / len(head)) > 0.01:  # >1% NUL bytes
        return False
    try:
        head.decode("utf-8")
        return True
    except Exception:
        try:
            head.decode("latin-1")
            return False
        except Exception:
            return False


def _suggest_handler_from_ext_and_mime(ext: Optional[str], mime: Optional[str]) -> str:
    """
    Suggest a coarse handler key from extension and mime.
    Returns one of: 'txt', 'pdf', 'docx', 'archive', 'image', 'binary'
    """
    ext = (ext or "").lower()
    mime = (mime or "").lower()
    if ext in TEXT_EXTS:
        return "txt"
    if ext in PDF_EXTS or mime == "application/pdf":
        return "pdf"
    if ext in DOCX_EXTS:
        return "docx"
    if ext in ARCHIVE_EXTS or ("zip" in (mime or "")):
        return "archive"
    if ext in IMAGE_EXTS or (mime.startswith("image/")):
        return "image"
    if mime.startswith("text/"):
        return "txt"
    return "binary"


def _write_filelike_to_temp(filelike: io.IOBase, filename_hint: Optional[str], cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Persist a file-like object to a temporary file under cache_dir (or system tmp dir) and return the Path.
    - This simplifies integration with libraries that require file paths (PyMuPDF, pypdf, python-docx).
    - Caller should ensure cache_dir exists and handle cleanup policy.
    """
    parent = Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename_hint).suffix if filename_hint else ""
    with tempfile.NamedTemporaryFile(delete=False, dir=str(parent), suffix=suffix) as tmp:
        # try to reset to beginning when possible
        try:
            filelike.seek(0)
        except Exception:
            pass
        while True:
            chunk = filelike.read(64 * 1024)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", errors="replace")
            tmp.write(chunk)
        tmp.flush()
        tmp_path = Path(tmp.name)
    return tmp_path


# ---------------------------
# Data contracts (lightweight)
# ---------------------------


def file_descriptor_template() -> Dict[str, Any]:
    """
    Returns the canonical FileDescriptor shape (helpful for tests and docs).
    """
    return {
        "source": None,  # original input: path string or upload identifier
        "path": None,  # Path on disk if available (temp path for uploads)
        "file_like": None,  # original file-like if not written to disk
        "ext": None,  # lower-case extension
        "mime": None,  # optional sniffer result
        "size": None,
        "mtime": None,
        "quick_hash": None,
        "suggested_handler": None,
        "notes": [],
    }


def document_template() -> Dict[str, Any]:
    """
    Document contract produced by loader/handlers (raw extraction stage).
    Document is minimal: text or image, id, and metadata for provenance.
    """
    return {
        "id": None,
        "raw_text": None,  # may be None if image-only and OCR not run here
        "image_bytes": None,  # optional; only for image pages or when rendering used
        "metadata": {
            "source_path": None,
            "file_type": None,
            "page_number": None,
            "file_hash": None,
            "page_hash": None,
            "extraction_engine": None,
            "ocr_engine": None,
            "ocr_confidence": None,
            "ingested_at": None,
            "offsets": None,
        },
    }


# ---------------------------
# Handler base classes & stubs
# ---------------------------


class Handler(abc.ABC):
    """
    Abstract handler base class.

    Concrete handlers should implement:
    - inspect(file_descriptor) -> dict (file-level metadata like page_count, has_embedded_text)
    - extract(file_descriptor) -> yields page-level extractions (page_number, raw_text, image_bytes, meta)

    Handler implementations should not perform cleaning; they produce raw extraction only.
    """

    @abc.abstractmethod
    def inspect(self, fd: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quick inspection without heavy work.
        Example outputs:
          { "page_count": 3, "has_embedded_text": True, "languages": ["en"] }
        """
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, fd: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Yield page-level dicts:
          { "page_number": int, "raw_text": Optional[str], "image_bytes": Optional[bytes], "meta": {...} }
        Extraction must be streaming where possible to avoid reading entire file into memory.
        """
        raise NotImplementedError


# Concrete handler stubs: implement these gradually


class TxtHandler(Handler):
    """
    Handler for plain text files.
    Implementation notes:
    - Prefer opening in binary and decoding in chunks using detected encoding (charset-normalizer or chardet).
    - Stream lines to avoid memory spikes for large text files.
    - Yield a single 'page' or split into multi-line pages based on config (e.g., N lines per document).
    """

    def inspect(self, fd: Dict[str, Any]) -> Dict[str, Any]:
        # Minimal quick inspection for text: page_count = 1, has_embedded_text True
        return {"page_count": 1, "has_embedded_text": True}

    def extract(self, fd: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Example simple behaviour:
        - If fd['path'] exists: open and yield a single page with raw_text = file content (or chunked).
        - If fd['file_like'] provided and path not set: read from file_like (seek to 0 if possible).
        """
        # TODO: replace this simple example with your streaming/chunking policy.
        path = fd.get("path")
        filelike = fd.get("file_like")
        head = b""
        if path:
            try:
                with open(path, "rb") as f:
                    head = f.read()
                    try:
                        raw = head.decode("utf-8")
                    except Exception:
                        raw = head.decode("latin-1", errors="replace")
                    yield {"page_number": 1, "raw_text": raw, "image_bytes": None, "meta": {}}
            except Exception as e:
                logger.exception("TxtHandler failed to read path %s: %s", path, e)
                return
        elif filelike:
            try:
                try:
                    filelike.seek(0)
                except Exception:
                    pass
                data = filelike.read()
                if isinstance(data, bytes):
                    try:
                        raw = data.decode("utf-8")
                    except Exception:
                        raw = data.decode("latin-1", errors="replace")
                else:
                    raw = str(data)
                yield {"page_number": 1, "raw_text": raw, "image_bytes": None, "meta": {}}
            except Exception as e:
                logger.exception("TxtHandler failed to read filelike: %s", e)
                return
        else:
            return


class PdfHandler(Handler):
    """
    PDF handler (text-first strategy).
    Implementation plan:
    - inspect(): try to use a light-weight PDF library to return page_count and whether pages have embedded text
    - extract(): iterate pages:
        * If embedded text present for page -> yield raw_text
        * Else -> yield image_bytes (rendered page) with meta indicating OCR fallback needed
    Notes:
    - Many PDF libraries require a filesystem path; if so, ensure discover wrote uploads to temp files.
    - Keep heavy rendering and OCR out of extract() unless configured to run OCR.
    """

    def inspect(self, fd: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: use pypdf or fitz (PyMuPDF) to inspect PDF quickly
        return {"page_count": None, "has_embedded_text": None}

    def extract(self, fd: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        # TODO: implement text-first extraction and yield page dicts
        # Example yields:
        # yield {"page_number": 1, "raw_text": "some text", "image_bytes": None, "meta": {"engine": "pypdf"}}
        # or if page empty:
        # yield {"page_number": 2, "raw_text": None, "image_bytes": b"...", "meta": {"rendered": True}}
        return
        yield  # make it a generator (no-op)


class DocxHandler(Handler):
    """
    Office handler for DOCX/PPTX/XLSX where appropriate.
    Notes:
    - Use python-docx for docx paragraphs
    - For pptx, use python-pptx to extract text per slide
    - Yield sensible page_number / paragraph indexing for provenance
    """

    def inspect(self, fd: Dict[str, Any]) -> Dict[str, Any]:
        return {"page_count": None, "has_embedded_text": True}

    def extract(self, fd: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        # TODO: implement docx/pptx extraction
        return
        yield


class ImageHandler(Handler):
    """
    Image handler: returns image_bytes for OCR downstream.
    - Do not run OCR here unless explicitly configured. Just yield the bytes and minimal metadata.
    """

    def inspect(self, fd: Dict[str, Any]) -> Dict[str, Any]:
        return {"page_count": 1, "has_embedded_text": False}

    def extract(self, fd: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        path = fd.get("path")
        filelike = fd.get("file_like")
        if path:
            try:
                b = path.read_bytes()
                yield {"page_number": 1, "raw_text": None, "image_bytes": b, "meta": {}}
            except Exception as e:
                logger.exception("ImageHandler failed reading path %s: %s", path, e)
                return
        elif filelike:
            try:
                try:
                    filelike.seek(0)
                except Exception:
                    pass
                data = filelike.read()
                if isinstance(data, str):
                    data = data.encode("utf-8", errors="replace")
                yield {"page_number": 1, "raw_text": None, "image_bytes": data, "meta": {}}
            except Exception as e:
                logger.exception("ImageHandler failed reading filelike: %s", e)
                return
        else:
            return


class ArchiveHandler(Handler):
    """
    Archive handler: iterate entries inside a zip/tar and delegate each entry to the appropriate handler.
    Important: track compound source_path like "archive.zip::folder/file.pdf" for provenance.
    """

    def inspect(self, fd: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: implement archive inspection (list entries)
        return {"page_count": None, "has_embedded_text": None}

    def extract(self, fd: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        # TODO: iterate entries, create nested file descriptors, and yield page extractions
        return
        yield


# ---------------------------
# OCR client and cache stub
# ---------------------------


class OcrClient:
    """
    Minimal OCR client interface.
    Implementations should provide .process(image_bytes, lang_hint=None) returning:
      { "text": str, "boxes": optional list, "confidence": float, "engine": "tesseract" or "gcloud", "meta": {...} }

    Concrete implementations should also expose engine version for provenance.
    """

    def __init__(self, provider: str = "local", config: Optional[Dict[str, Any]] = None):
        self.provider = provider
        self.config = config or {}

    def process(self, image_bytes: bytes, lang_hint: Optional[str] = None) -> Dict[str, Any]:
        # TODO: integrate with pytesseract or cloud APIs
        # For now, raise to indicate not implemented
        raise NotImplementedError("OCR client not implemented")


# ---------------------------
# Status/manifest store stub
# ---------------------------


class StatusStore:
    """
    Simple status store interface for per-file processing metadata and idempotency.
    Implementations could be:
      - jsonl files under cache_dir/manifests
      - sqlite DB (recommended for scale)
      - Redis (if distributed)
    Minimal required methods:
      - get_status(file_hash) -> dict or None
      - mark_processed(file_hash, details)
      - mark_failed(file_hash, details)
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # TODO: implement persistent backing store
        self._store: Dict[str, Dict[str, Any]] = {}

    def get_status(self, file_hash: str) -> Optional[Dict[str, Any]]:
        return self._store.get(file_hash)

    def mark_processed(self, file_hash: str, details: Dict[str, Any]):
        self._store[file_hash] = {"state": "processed", "details": details, "timestamp": int(time.time())}

    def mark_failed(self, file_hash: str, details: Dict[str, Any]):
        self._store[file_hash] = {"state": "failed", "details": details, "timestamp": int(time.time())}


# ---------------------------
# Loader orchestrator
# ---------------------------


class Loader:
    """
    Loader orchestrates discovery and extraction.

    Responsibilities in methods below:
    - discover: implemented as a solid, usable function (it reuses helper utilities).
    - choose_handler: pick a Handler instance for a FileDescriptor.
    - process_file: orchestrate per-file work, yield Document dicts.
    - stream_documents: top-level convenience to process many inputs.
    - preview: return first N document samples for a file (dry-run friendly).
    """

    def __init__(
        self,
        *,
        cache_dir: Optional[Union[str, Path]] = None,
        handlers: Optional[Dict[str, Handler]] = None,
        ocr_client: Optional[OcrClient] = None,
        status_store: Optional[StatusStore] = None,
        ignore_patterns: Optional[Iterable[str]] = None,
        recursive: bool = RECURSIVE_BY_DEFAULT,
    ):
        """
        Setup loader state and handler registry.

        Arguments:
        - cache_dir: local dir for temp files, OCR cache, manifests
        - handlers: mapping of handler_key -> Handler instance (e.g., "txt": TxtHandler())
        - ocr_client: optional OCR client for fallback
        - status_store: store for processed file metadata
        - ignore_patterns: list of glob patterns to skip
        - recursive: whether to walk directories recursively
        """
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.handlers = handlers or {}
        # ensure default handlers are present
        self.handlers.setdefault("txt", TxtHandler())
        self.handlers.setdefault("pdf", PdfHandler())
        self.handlers.setdefault("docx", DocxHandler())
        self.handlers.setdefault("image", ImageHandler())
        self.handlers.setdefault("archive", ArchiveHandler())
        self.ocr_client = ocr_client
        self.status_store = status_store or StatusStore(cache_dir=self.cache_dir)
        self.ignore_patterns = list(ignore_patterns) if ignore_patterns else IGNORE_PATTERNS_DEFAULT
        self.recursive = recursive

    # -----------------------
    # discover (implemented)
    # -----------------------
    def discover(
        self,
        input_path_or_filelike: Union[str, Path, Iterable[Union[str, Path, Any]], Any],
        *,
        recursive: Optional[bool] = None,
        ignore_patterns: Optional[Iterable[str]] = None,
        quick_hash_window: int = QUICK_HASH_WINDOW,
    ) -> Iterator[Dict[str, Any]]:
        """
        Discover files and yield canonical FileDescriptor dictionaries.

        This method handles:
        - single paths, lists of paths, glob patterns, directories
        - file-like upload objects (writes to a temp path in cache_dir)
        - ignore patterns and recursive directory walking

        Best practice: call this with a path, then pass each descriptor to process_file().
        """
        recursive = self.recursive if recursive is None else recursive
        ignore_patterns = list(ignore_patterns) if ignore_patterns is not None else self.ignore_patterns

        items: Iterable[Any]
        if input_path_or_filelike is None:
            return
        if isinstance(input_path_or_filelike, (str, Path)) or _is_file_like(input_path_or_filelike):
            items = [input_path_or_filelike]
        elif isinstance(input_path_or_filelike, Iterable):
            items = input_path_or_filelike  # type: ignore
        else:
            items = [input_path_or_filelike]

        for item in items:
            # Handle glob strings quickly
            try:
                if isinstance(item, str) and any(ch in item for ch in ["*", "?", "["]):
                    for p in Path().glob(item):
                        if any(fnmatch.fnmatch(str(p.name), pat) for pat in ignore_patterns):
                            continue
                        if p.is_dir():
                            for sub in p.rglob("*") if recursive else p.glob("*"):
                                if sub.is_file():
                                    yield from self.discover(sub, recursive=recursive, ignore_patterns=ignore_patterns, quick_hash_window=quick_hash_window)
                            continue
                        yield from self.discover(p, recursive=recursive, ignore_patterns=ignore_patterns, quick_hash_window=quick_hash_window)
                    continue
            except Exception as e:
                logger.exception("Error expanding glob %s: %s", item, e)

            # Path-like handling
            if isinstance(item, (str, Path)) and not _is_file_like(item):
                path = Path(item)
                if path.is_dir():
                    for sub in path.rglob("*") if recursive else path.glob("*"):
                        if not sub.is_file():
                            continue
                        if any(fnmatch.fnmatch(str(sub.name), pat) for pat in ignore_patterns):
                            continue
                        yield from self.discover(sub, recursive=recursive, ignore_patterns=ignore_patterns, quick_hash_window=quick_hash_window)
                    continue

                fd = file_descriptor_template()
                fd["source"] = str(item)
                fd["path"] = path
                fd["ext"] = path.suffix.lower()
                fd["notes"] = []
                try:
                    try:
                        stat = path.stat()
                        fd["size"] = int(stat.st_size)
                        fd["mtime"] = int(stat.st_mtime)
                    except Exception:
                        fd["notes"].append("stat_failed")

                    head = _read_head_bytes_from_path(path, MIME_SNIFF_BYTES)
                    fd["mime"] = _sniff_mime(head, path=path)
                    fd["quick_hash"] = _quick_hash_of(path=path, filelike=None, window=quick_hash_window)
                    fd["notes"].append("utf_likely" if _utf_likely_from_bytes(head) else "non_utf_likely")
                    fd["suggested_handler"] = _suggest_handler_from_ext_and_mime(fd["ext"], fd.get("mime", ""))
                except Exception as e:
                    logger.exception("Error inspecting path %s: %s", item, e)
                    fd["notes"].append(f"inspect_error:{e!s}")
                yield fd
                continue

            # File-like (upload) handling
            if _is_file_like(item):
                filelike = item
                filename_hint = getattr(filelike, "filename", None) or getattr(filelike, "name", None) or "upload"
                fd = file_descriptor_template()
                fd["source"] = getattr(filelike, "filename", None) or "<file-like>"
                fd["file_like"] = filelike
                fd["ext"] = Path(filename_hint).suffix.lower()
                fd["notes"].append("uploaded")
                try:
                    head = _read_head_bytes_from_filelike(filelike, MIME_SNIFF_BYTES)
                    fd["mime"] = _sniff_mime(head, path=Path(filename_hint) if filename_hint else None)
                    # try to guess size if seekable
                    try:
                        filelike.seek(0, os.SEEK_END)
                        fd["size"] = filelike.tell()
                        filelike.seek(0)
                    except Exception:
                        fd["size"] = None
                    fd["quick_hash"] = _quick_hash_of(path=None, filelike=filelike, window=quick_hash_window)
                    fd["notes"].append("utf_likely" if _utf_likely_from_bytes(head) else "non_utf_likely")
                    fd["suggested_handler"] = _suggest_handler_from_ext_and_mime(fd["ext"], fd.get("mime", ""))
                    # write to temp path to simplify downstream libraries that require a path
                    try:
                        tmp_path = _write_filelike_to_temp(filelike, filename_hint, cache_dir=self.cache_dir)
                        fd["path"] = tmp_path
                        fd["notes"].append("temp_written")
                        try:
                            stat = tmp_path.stat()
                            fd["size"] = int(stat.st_size)
                            fd["mtime"] = int(stat.st_mtime)
                        except Exception:
                            pass
                    except Exception as e:
                        logger.exception("Failed to write upload to temp file: %s", e)
                        fd["notes"].append(f"temp_write_failed:{e!s}")
                except Exception as e:
                    logger.exception("Error inspecting upload: %s", e)
                    fd["notes"].append(f"upload_inspect_error:{e!s}")
                yield fd
                continue

            # Fallback unknown type
            try:
                yield {
                    "source": repr(item),
                    "path": None,
                    "file_like": None,
                    "ext": None,
                    "mime": None,
                    "size": None,
                    "mtime": None,
                    "quick_hash": None,
                    "suggested_handler": None,
                    "notes": ["unhandled_input_type"],
                }
            except Exception:
                continue

    # end discover

    # -----------------------
    # remaining orchestrator methods (skeletons with rich comments)
    # -----------------------

    def choose_handler(self, fd: Dict[str, Any]) -> Handler:
        """
        Decide which Handler instance to use for this FileDescriptor.

        Strategy:
        1. Use fd['suggested_handler'] if present and self.handlers contains it.
        2. Otherwise, use mime-based rules to pick a handler.
        3. Fallback to a generic 'binary' or 'txt' handler depending on utf heuristic.
        4. Raise or return a default handler if none match.

        Notes:
        - Keep this method deterministic (no heavy I/O).
        - Do not run OCR or rendering here.
        """
        suggested = fd.get("suggested_handler")
        if suggested and suggested in self.handlers:
            return self.handlers[suggested]
        mime = (fd.get("mime") or "").lower()
        ext = (fd.get("ext") or "").lower()
        # coarse rules
        if ext in TEXT_EXTS or mime.startswith("text"):
            return self.handlers["txt"]
        if ext in PDF_EXTS or "pdf" in mime:
            return self.handlers["pdf"]
        if ext in DOCX_EXTS:
            return self.handlers["docx"]
        if ext in IMAGE_EXTS or (mime.startswith("image")):
            return self.handlers["image"]
        if ext in ARCHIVE_EXTS or "zip" in mime:
            return self.handlers["archive"]
        # fallback
        return self.handlers.get("txt", list(self.handlers.values())[0])

    def _make_document_id(self, fd: Dict[str, Any], page_number: int, chunk_index: Optional[int] = None) -> str:
        """
        Helper for stable document ids. Example scheme:
          {file_quick_hash}::p{page_number}::c{chunk_index}
        Ensure id is deterministic based on file quick_hash and page index.
        """
        file_hash = fd.get("quick_hash") or "nofhash"
        if chunk_index is not None:
            return f"{file_hash}::p{page_number}::c{chunk_index}"
        return f"{file_hash}::p{page_number}"

    def process_file(self, fd: Dict[str, Any], *, preview: bool = False, ocr_enabled: bool = True, force: bool = False) -> Iterator[Dict[str, Any]]:
        """
        Orchestrate extraction for a single FileDescriptor and yield Document dicts.

        High-level plan:
        - Check status_store for existing processed state: skip unless force.
        - Call choose_handler(fd) to get a handler instance.
        - Call handler.inspect(fd) to get file-level metadata.
        - Iterate handler.extract(fd):
            - For each page: compute page_hash
            - If page has raw_text: create Document with raw_text and metadata, yield
            - If page has image_bytes and ocr_enabled:
                * check OCR cache (not implemented here)
                * call self.ocr_client.process(image_bytes) if available
                * yield Document with ocr text + ocr meta
            - If page has image_bytes and OCR disabled: yield Document with image_bytes and metadata (cleaner or later stage may decide)
        - Update status_store.mark_processed on success or mark_failed on exceptions.
        - If preview True: stop after N pages (implement preview count outside this method).
        - On exceptions, yield nothing and mark file failed with reason.

        Implementation is left as skeleton so you can implement carefully and test each step.
        """
        # TODO: check status store (skip if already processed and not force).
        # TODO: choose handler and call inspect. Log results.
        # TODO: iterate handler.extract and produce Documents:
        #   - compute page_hash (sha256 of raw_text or image bytes + page_number)
        #   - build metadata dictionary using document_template()['metadata']
        #   - set extraction_engine in metadata (e.g., 'pypdf' or 'pymupdf')
        #   - if OCR needed and ocr_client is present: call ocr_client.process and attach results
        #   - assign stable id using _make_document_id
        #   - yield document dicts
        # TODO: update status_store at end (mark_processed) with summary (pages_count, bytes, runtime)
        # NOTE: keep this method streaming: do not accumulate all pages in memory.
        raise NotImplementedError("process_file orchestration must be implemented by developer")

    def stream_documents(self, inputs: Union[str, Path, Iterable[Any], Any], *, preview: bool = False, ocr_enabled: bool = True, force: bool = False) -> Iterator[Dict[str, Any]]:
        """
        Convenience top-level method:
        - Calls discover(inputs) to get FileDescriptors
        - For each descriptor, call process_file(fd) and yield all Documents
        - Collect and log basic metrics (files processed, pages extracted)

        Keep this method thin so you can call it from CLI or tests.
        """
        # TODO: iterate discover(inputs) and for each descriptor call process_file(fd)
        # Emit high-level logs and metrics, but keep actual extraction in process_file
        raise NotImplementedError("stream_documents must be implemented by developer")

    def preview(self, input_item: Union[str, Path, Any], n_pages: int = 2) -> List[Dict[str, Any]]:
        """
        Extract first n_pages from the first file discovered for quick debugging.

        Behaviour:
        - Call discover(input_item) and pick the first descriptor
        - Call process_file(fd, preview=True) and collect up to n_pages documents
        - Return list of document dicts (do not persist status_store changes in preview mode)
        """
        # TODO: implement by wiring discover() + process_file() with a preview flag
        raise NotImplementedError("preview must be implemented")

    def mark_processed(self, fd: Dict[str, Any], details: Dict[str, Any]):
        """
        Convenience wrapper to mark file processed in status_store.
        details should contain summary info such as pages_count, bytes_processed, pipeline_tag.
        """
        file_hash = fd.get("quick_hash")
        if not file_hash:
            # fall back to path-based hash if needed
            file_hash = hashlib.sha256(str(fd.get("source")).encode()).hexdigest()
        self.status_store.mark_processed(file_hash, details)

    def get_status(self, fd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get stored processing status for a file descriptor (if any).
        """
        file_hash = fd.get("quick_hash")
        if not file_hash:
            file_hash = hashlib.sha256(str(fd.get("source")).encode()).hexdigest()
        return self.status_store.get_status(file_hash)


# End of loader.py
