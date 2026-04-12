"""S3 helpers for RunPod workers.

The handlers move files between S3 and the worker's ephemeral disk. All
helpers accept ``s3://bucket/key`` URIs; callers never construct boto3 objects
directly. A single module-level client is lazily created so that the first
handler invocation pays the boto3 initialization cost once.
"""
import logging
import os
import tempfile
from typing import Optional, Tuple

import boto3

logger = logging.getLogger(__name__)

_client = None


def _s3():
    """Return a module-level boto3 S3 client (created on first call)."""
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2"),
        )
    return _client


def parse_uri(uri: str) -> Tuple[str, str]:
    """Split an ``s3://bucket/key`` URI into ``(bucket, key)``."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3:// URI: {uri}")
    rest = uri[5:]
    if "/" not in rest:
        raise ValueError(f"S3 URI missing key: {uri}")
    bucket, key = rest.split("/", 1)
    return bucket, key


def download_file(uri: str, local_path: str) -> str:
    """Download a single object to ``local_path`` (parent dirs auto-created)."""
    bucket, key = parse_uri(uri)
    parent = os.path.dirname(local_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    _s3().download_file(bucket, key, local_path)
    return local_path


def download_to_temp(uri: str, suffix: Optional[str] = None) -> str:
    """Download an object to a named tempfile and return its local path.

    ``suffix`` defaults to the source filename's extension so downstream code
    that inspects extensions (e.g. soundfile, ffmpeg) works as expected.
    """
    bucket, key = parse_uri(uri)
    if suffix is None:
        name = key.rsplit("/", 1)[-1]
        suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    _s3().download_file(bucket, key, path)
    return path


def download_prefix(prefix_uri: str, local_dir: str) -> int:
    """Mirror everything under ``prefix_uri`` to ``local_dir``.

    Returns the number of files downloaded. Existing local files are
    overwritten — the worker disk is ephemeral so this is always safe.
    """
    bucket, prefix = parse_uri(prefix_uri)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    os.makedirs(local_dir, exist_ok=True)

    paginator = _s3().get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix):]
            if not rel:
                continue
            dst = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dst) or local_dir, exist_ok=True)
            _s3().download_file(bucket, key, dst)
            count += 1
    logger.info(f"[s3] downloaded {count} files from {prefix_uri} -> {local_dir}")
    return count


def ensure_local(local_dir: str, s3_fallback_uri: str, marker: str = ".s3_synced") -> str:
    """Ensure ``local_dir`` exists and has content; download from S3 if empty.

    On a fresh network volume the directory will be missing.  This function
    downloads the pretrained assets from ``s3_fallback_uri`` *once*, then
    writes a small marker file so subsequent cold starts skip the download.

    Returns ``local_dir`` unchanged (convenience for chaining).
    """
    marker_path = os.path.join(local_dir, marker)
    if os.path.isdir(local_dir) and os.path.exists(marker_path):
        return local_dir

    logger.info(f"[s3] ensure_local: {local_dir} missing or empty, syncing from {s3_fallback_uri}")
    count = download_prefix(s3_fallback_uri, local_dir)
    if count > 0:
        with open(marker_path, "w") as f:
            f.write(f"synced {count} files from {s3_fallback_uri}\n")
        logger.info(f"[s3] ensure_local: synced {count} files to {local_dir}")
    else:
        logger.warning(f"[s3] ensure_local: 0 files downloaded from {s3_fallback_uri}")
    return local_dir


def upload_file(local_path: str, uri: str, content_type: Optional[str] = None) -> str:
    """Upload a single file. Returns the same ``uri`` on success."""
    bucket, key = parse_uri(uri)
    extra = {"ContentType": content_type} if content_type else None
    _s3().upload_file(local_path, bucket, key, ExtraArgs=extra)
    return uri


def upload_dir(local_dir: str, prefix_uri: str) -> int:
    """Recursively upload every file in ``local_dir`` under ``prefix_uri``.

    The relative path from ``local_dir`` is preserved in S3. Returns the
    number of files uploaded.
    """
    bucket, prefix = parse_uri(prefix_uri)
    prefix = prefix.rstrip("/")
    count = 0
    for root, _, files in os.walk(local_dir):
        for name in files:
            local_path = os.path.join(root, name)
            rel = os.path.relpath(local_path, local_dir).replace(os.sep, "/")
            key = f"{prefix}/{rel}" if prefix else rel
            _s3().upload_file(local_path, bucket, key)
            count += 1
    logger.info(f"[s3] uploaded {count} files from {local_dir} -> {prefix_uri}")
    return count
