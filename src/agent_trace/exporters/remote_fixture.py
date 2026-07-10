"""
Durable/remote fixture backend — so a recording survives a killed or swept
worker process.

Background
----------

Fixtures and trace files are written only to the invoking process's local
filesystem (``Tracer.start_trace()``'s ``run_dir``); the only non-local
exporter is OTLP, which carries span metadata but not HTTP request/response
bodies. On managed platforms (e.g. LangGraph Cloud — issue #7417), a worker
whose run gets swept/re-dispatched (or simply killed) has an ephemeral,
developer-inaccessible local filesystem — any fixture recorded there is
unrecoverable after the fact.

This module adds an object-store-backed ``RemoteFixtureBackend`` so a
recorded trace and its captured wire-level bodies persist independently of
the worker process that produced them:

  - :class:`RemoteFixtureBackend` — the storage-agnostic protocol
    (``put_bytes``/``get_bytes``/``list_keys``).
  - :class:`LocalDirRemoteFixtureBackend` — writes to a second directory,
    e.g. a network-mounted (NFS/EFS/SMB) path distinct from the worker's own
    ephemeral local disk. A fully real, dependency-free backend on its own
    for any deployment where a shared mount is available, and the reference
    implementation the two cloud backends below are tested against.
  - :class:`S3RemoteFixtureBackend` — lazy ``import boto3``; requires
    ``pip install boto3``.
  - :class:`GCSRemoteFixtureBackend` — lazy
    ``import google.cloud.storage``; requires
    ``pip install google-cloud-storage``.

Usage — durable per-exchange sync while recording (survives the worker
being killed mid-run, not just at clean exit)::

    from agent_trace.exporters.remote_fixture import (
        LocalDirRemoteFixtureBackend,
        remote_sync_callback,
    )

    backend = LocalDirRemoteFixtureBackend(Path("/mnt/shared/agent-trace"))
    with tracer.start_trace("my_graph", record=True,
                             remote_backend=backend) as trace:
        ...

Usage — one-shot sync/restore of an already-recorded run::

    from agent_trace.exporters.remote_fixture import (
        sync_run_to_remote,
        restore_run_from_remote,
    )

    sync_run_to_remote(run_dir, backend, run_id="run_abc123")
    restored_dir = restore_run_from_remote(backend, run_id="run_abc123",
                                            dest_dir=Path("/tmp/restored"))
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

__all__ = [
    "GCSRemoteFixtureBackend",
    "LocalDirRemoteFixtureBackend",
    "RemoteFixtureBackend",
    "S3RemoteFixtureBackend",
    "remote_sync_callback",
    "restore_run_from_remote",
    "sync_run_to_remote",
]

logger = logging.getLogger(__name__)

_S3_INSTALL_HINT = (
    "S3RemoteFixtureBackend requires boto3.\nInstall it with:\n\n"
    "    pip install boto3\n"
)
_GCS_INSTALL_HINT = (
    "GCSRemoteFixtureBackend requires google-cloud-storage.\n"
    "Install it with:\n\n"
    "    pip install google-cloud-storage\n"
)


class RemoteFixtureBackend(ABC):
    """Storage-agnostic protocol for durably persisting fixture data
    somewhere other than the worker process's own local filesystem."""

    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> None:
        """Durably store *data* under *key*, overwriting any existing value."""

    @abstractmethod
    def get_bytes(self, key: str) -> bytes | None:
        """Return the bytes stored under *key*, or None if absent."""

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]:
        """Return every stored key starting with *prefix*."""


class LocalDirRemoteFixtureBackend(RemoteFixtureBackend):
    """Writes to a second directory — e.g. a network-mounted path distinct
    from the worker's own ephemeral local disk. Real and fully functional
    (not a mock/stub): a legitimate deployment choice on its own whenever a
    shared mount (NFS/EFS/SMB) is available, and the backend the cloud
    implementations below are tested against for correctness."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        base = self.root.resolve()
        candidate = (base / key).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            raise ValueError(f"Invalid key {key!r}: path traversal detected") from None
        return candidate

    def put_bytes(self, key: str, data: bytes) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get_bytes(self, key: str) -> bytes | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        return path.read_bytes()

    def list_keys(self, prefix: str) -> list[str]:
        base = self.root.resolve()
        prefix_path = self._path_for(prefix)
        search_root = prefix_path if prefix_path.is_dir() else prefix_path.parent
        if not search_root.exists():
            return []
        return sorted(
            str(p.relative_to(base))
            for p in search_root.rglob("*")
            if p.is_file() and str(p.relative_to(base)).startswith(prefix)
        )


class S3RemoteFixtureBackend(RemoteFixtureBackend):
    """S3-backed remote fixture store. ``boto3`` is imported lazily — this
    class can be referenced (e.g. for isinstance checks) even when boto3
    isn't installed; only constructing an instance requires it."""

    def __init__(self, bucket: str, prefix: str = "", client: Any = None) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self._client = client or self._make_client()

    def _make_client(self) -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(_S3_INSTALL_HINT) from exc
        return boto3.client("s3")

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=self._full_key(key), Body=data)

    def get_bytes(self, key: str) -> bytes | None:
        try:
            response = self._client.get_object(
                Bucket=self.bucket, Key=self._full_key(key)
            )
        except Exception:
            # botocore raises a dynamically-generated ClientError subclass
            # (e.g. NoSuchKey) — catching Exception broadly here (rather
            # than importing botocore just to catch its specific error
            # type) keeps this backend usable with any S3-compatible client
            # a caller supplies via the `client=` param.
            return None
        return response["Body"].read()  # type: ignore[no-any-return]

    def list_keys(self, prefix: str) -> list[str]:
        full_prefix = self._full_key(prefix)
        keys: list[str] = []
        continuation_token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": full_prefix}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            response = self._client.list_objects_v2(**kwargs)
            for obj in response.get("Contents", []):
                remote_key = obj["Key"]
                keys.append(
                    remote_key[len(self.prefix) + 1 :]
                    if self.prefix
                    else remote_key
                )
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
        return sorted(keys)


class GCSRemoteFixtureBackend(RemoteFixtureBackend):
    """Google Cloud Storage-backed remote fixture store.
    ``google-cloud-storage`` is imported lazily."""

    def __init__(self, bucket: str, prefix: str = "", client: Any = None) -> None:
        self.prefix = prefix.rstrip("/")
        self._bucket = client or self._make_bucket(bucket)

    def _make_bucket(self, bucket_name: str) -> Any:
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise ImportError(_GCS_INSTALL_HINT) from exc
        return storage.Client().bucket(bucket_name)

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_bytes(self, key: str, data: bytes) -> None:
        blob = self._bucket.blob(self._full_key(key))
        blob.upload_from_string(data)

    def get_bytes(self, key: str) -> bytes | None:
        blob = self._bucket.blob(self._full_key(key))
        if not blob.exists():
            return None
        return blob.download_as_bytes()  # type: ignore[no-any-return]

    def list_keys(self, prefix: str) -> list[str]:
        full_prefix = self._full_key(prefix)
        blobs = self._bucket.list_blobs(prefix=full_prefix)
        keys = [
            blob.name[len(self.prefix) + 1 :] if self.prefix else blob.name
            for blob in blobs
        ]
        return sorted(keys)


# ---------------------------------------------------------------------------
# Sync helpers
# ---------------------------------------------------------------------------


def remote_sync_callback(
    backend: RemoteFixtureBackend, run_id: str
) -> Any:
    """Return an ``on_exchange_recorded`` callback (see ``Fixture.__init__``)
    that durably uploads each exchange to *backend* as it's recorded — so a
    worker killed mid-run still has every exchange recorded up to that
    point recoverable from remote storage, not just whatever made it into
    the local, possibly-never-read-again ``fixture.db``.

    Best-effort by design (matching Fixture's own on_exchange_recorded
    contract): swallows every exception itself so a remote upload failure
    can never break local recording.
    """

    def _callback(exchange: dict[str, Any]) -> None:
        try:
            key = f"{run_id}/exchanges/{exchange['sequence_num']:08d}.json"
            backend.put_bytes(key, json.dumps(exchange).encode("utf-8"))
        except Exception:
            logger.warning(
                "agent-trace: failed to sync exchange %s to remote backend",
                exchange.get("sequence_num"),
                exc_info=True,
            )

    return _callback


def sync_run_to_remote(
    run_dir: Path, backend: RemoteFixtureBackend, run_id: str
) -> None:
    """Upload ``run_dir/fixture.db`` and ``run_dir/trace.json`` (whichever
    exist) to *backend* under ``{run_id}/fixture.db`` /
    ``{run_id}/trace.json`` — a one-shot sync of an already-recorded run,
    complementary to (not a replacement for) :func:`remote_sync_callback`'s
    per-exchange durability during recording."""
    for filename in ("fixture.db", "trace.json"):
        path = run_dir / filename
        if path.exists():
            backend.put_bytes(f"{run_id}/{filename}", path.read_bytes())


def restore_run_from_remote(
    backend: RemoteFixtureBackend, run_id: str, dest_dir: Path
) -> Path:
    """Download ``{run_id}/fixture.db`` / ``{run_id}/trace.json`` (whichever
    exist in *backend*) into ``dest_dir/run_id/``, returning that directory
    — the recovery path for a worker whose local filesystem is gone but
    whose recordings were synced to remote storage while it ran."""
    run_dir = dest_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("fixture.db", "trace.json"):
        data = backend.get_bytes(f"{run_id}/{filename}")
        if data is not None:
            (run_dir / filename).write_bytes(data)
    return run_dir
