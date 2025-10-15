from pathlib import Path
import io

import pytest

from data_ingestion.loader import Loader, TxtHandler, document_template


def test_process_file_txthandler_and_status(tmp_path):
    # create file
    p = tmp_path / "sample.txt"
    content = "first line\nsecond line\n"
    p.write_text(content, encoding="utf-8")

    loader = Loader(cache_dir=str(tmp_path))
    # get first fd from discover
    fd = next(loader.discover(str(p)))
    # ensure not preview, so status gets marked
    docs = list(loader.process_file(fd, preview=False, ocr_enabled=False, force=True))
    assert len(docs) == 1
    d = docs[0]
    assert d["raw_text"] is not None
    assert "first line" in d["raw_text"]
    assert d["metadata"]["source_path"].endswith("sample.txt")
    # status store should be updated
    status = loader.status_store.get_status(fd["quick_hash"])
    assert status is not None
    assert status["state"] == "processed"
    assert status["details"]["pages"] == 1


def test_preview_returns_pages(tmp_path):
    p = tmp_path / "many.txt"
    p.write_text("\n".join([f"line {i}" for i in range(10)]), encoding="utf-8")

    loader = Loader(cache_dir=str(tmp_path))
    previews = loader.preview(str(p), n_pages=1)
    assert isinstance(previews, list)
    assert len(previews) == 1
    assert "raw_text" in previews[0]


def test_stream_documents_multiple_files(tmp_path):
    # create two files
    p1 = tmp_path / "a.txt"
    p2 = tmp_path / "b.txt"
    p1.write_text("A", encoding="utf-8")
    p2.write_text("B", encoding="utf-8")

    loader = Loader(cache_dir=str(tmp_path))
    docs = list(loader.stream_documents([str(p1), str(p2)], preview=False, ocr_enabled=False, force=True))
    # two docs expected (one per file)
    assert len(docs) >= 2
    # ensure status store has entries for both
    fds = list(loader.discover(str(tmp_path)))
    for fd in fds:
        if fd["quick_hash"]:
            s = loader.status_store.get_status(fd["quick_hash"])
            assert s is not None
            assert s["state"] == "processed"
