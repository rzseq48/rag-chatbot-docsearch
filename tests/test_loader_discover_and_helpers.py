import io
import os
from pathlib import Path

import pytest

from data_ingestion.loader import Loader, file_descriptor_template


def test_discover_text_file(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("hello world\nthis is a test\n", encoding="utf-8")

    loader = Loader(cache_dir=str(tmp_path))
    fds = list(loader.discover(str(p)))
    assert len(fds) == 1
    fd = fds[0]
    assert fd["path"] == p
    assert fd["ext"] == ".txt"
    assert fd["quick_hash"] is not None
    assert any(n.startswith("utf_likely") or n.startswith("non_utf_likely") for n in fd["notes"])


def test_discover_filelike_writes_temp(tmp_path):
    data = b"some uploaded bytes\nline2\n"
    filelike = io.BytesIO(data)
    # simulate an upload filename attribute (like Starlette UploadFile)
    filelike.filename = "upload.md"

    loader = Loader(cache_dir=str(tmp_path))
    fds = list(loader.discover(filelike))
    assert len(fds) == 1
    fd = fds[0]
    assert fd["file_like"] is filelike
    assert fd["path"] is not None
    assert Path(fd["path"]).exists()
    assert "temp_written" in fd["notes"]


def test_choose_handler_basic_rules():
    loader = Loader()
    fd = file_descriptor_template()
    fd["ext"] = ".txt"
    fd["mime"] = "text/plain"
    handler = loader.choose_handler(fd)
    # TxtHandler expected for text ext
    assert handler.__class__.__name__.lower().startswith("txt")


def test_discover_directory_recursive(tmp_path):
    # create a small directory tree
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.txt").write_text("1")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "two.txt").write_text("2")

    loader = Loader()
    results = list(loader.discover(tmp_path))
    # Expect two files
    assert any(str(r["path"]).endswith("one.txt") for r in results)
    assert any(str(r["path"]).endswith("two.txt") for r in results)
