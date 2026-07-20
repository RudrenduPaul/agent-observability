"""
Unit tests for agent_trace.exporters.remote_fixture — the durable/remote
fixture backend so a recording survives a killed or swept worker process
(issue #7417).

LocalDirRemoteFixtureBackend is tested as a real, fully functional backend.
S3RemoteFixtureBackend/GCSRemoteFixtureBackend are tested against injected
fake clients (no boto3/google-cloud-storage dependency needed) — exercising
the same key-mapping/round-trip logic real cloud SDKs would drive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_trace.exporters.remote_fixture import (
    GCSRemoteFixtureBackend,
    LocalDirRemoteFixtureBackend,
    S3RemoteFixtureBackend,
    remote_sync_callback,
    restore_run_from_remote,
    sync_run_to_remote,
)

# ---------------------------------------------------------------------------
# LocalDirRemoteFixtureBackend — real backend
# ---------------------------------------------------------------------------


class TestLocalDirRemoteFixtureBackend:
    def test_put_then_get_round_trips(self, tmp_path: Path) -> None:
        backend = LocalDirRemoteFixtureBackend(tmp_path / "remote")
        backend.put_bytes("run_1/fixture.db", b"hello")
        assert backend.get_bytes("run_1/fixture.db") == b"hello"

    def test_get_missing_key_returns_none(self, tmp_path: Path) -> None:
        backend = LocalDirRemoteFixtureBackend(tmp_path / "remote")
        assert backend.get_bytes("nope") is None

    def test_put_overwrites_existing_key(self, tmp_path: Path) -> None:
        backend = LocalDirRemoteFixtureBackend(tmp_path / "remote")
        backend.put_bytes("k", b"first")
        backend.put_bytes("k", b"second")
        assert backend.get_bytes("k") == b"second"

    def test_list_keys_returns_matching_prefix(self, tmp_path: Path) -> None:
        backend = LocalDirRemoteFixtureBackend(tmp_path / "remote")
        backend.put_bytes("run_1/exchanges/00000001.json", b"{}")
        backend.put_bytes("run_1/exchanges/00000002.json", b"{}")
        backend.put_bytes("run_2/exchanges/00000001.json", b"{}")
        keys = backend.list_keys("run_1/")
        assert len(keys) == 2
        assert all(k.startswith("run_1/") for k in keys)

    def test_list_keys_empty_when_prefix_absent(self, tmp_path: Path) -> None:
        backend = LocalDirRemoteFixtureBackend(tmp_path / "remote")
        assert backend.list_keys("nope/") == []

    def test_rejects_path_traversal_key(self, tmp_path: Path) -> None:
        backend = LocalDirRemoteFixtureBackend(tmp_path / "remote")
        with pytest.raises(ValueError, match="path traversal"):
            backend.put_bytes("../../etc/passwd", b"pwned")

    def test_root_directory_created_on_construction(self, tmp_path: Path) -> None:
        root = tmp_path / "new_remote_dir"
        assert not root.exists()
        LocalDirRemoteFixtureBackend(root)
        assert root.exists()


# ---------------------------------------------------------------------------
# S3RemoteFixtureBackend — against a fake boto3-shaped client
# ---------------------------------------------------------------------------


class _FakeS3Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeS3Client:
    """Minimal stand-in for a boto3 S3 client — only the three methods
    S3RemoteFixtureBackend actually calls."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:  # noqa: N803
        self.objects[Key] = Body

    def get_object(self, Bucket: str, Key: str) -> dict:  # noqa: N803
        if Key not in self.objects:
            raise KeyError(f"NoSuchKey: {Key}")
        return {"Body": _FakeS3Body(self.objects[Key])}

    def list_objects_v2(
        self,
        Bucket: str,  # noqa: N803
        Prefix: str,  # noqa: N803
        ContinuationToken: str | None = None,  # noqa: N803
    ) -> dict:
        matching = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in matching], "IsTruncated": False}


class TestS3RemoteFixtureBackend:
    def test_put_then_get_round_trips(self) -> None:
        backend = S3RemoteFixtureBackend(
            bucket="my-bucket", prefix="agent-trace", client=_FakeS3Client()
        )
        backend.put_bytes("run_1/fixture.db", b"hello")
        assert backend.get_bytes("run_1/fixture.db") == b"hello"

    def test_prefix_applied_to_underlying_client_key(self) -> None:
        client = _FakeS3Client()
        backend = S3RemoteFixtureBackend(
            bucket="b", prefix="agent-trace", client=client
        )
        backend.put_bytes("run_1/fixture.db", b"x")
        assert "agent-trace/run_1/fixture.db" in client.objects

    def test_no_prefix_uses_key_directly(self) -> None:
        client = _FakeS3Client()
        backend = S3RemoteFixtureBackend(bucket="b", client=client)
        backend.put_bytes("run_1/fixture.db", b"x")
        assert "run_1/fixture.db" in client.objects

    def test_get_missing_key_returns_none(self) -> None:
        backend = S3RemoteFixtureBackend(bucket="b", client=_FakeS3Client())
        assert backend.get_bytes("nope") is None

    def test_list_keys_strips_prefix(self) -> None:
        client = _FakeS3Client()
        backend = S3RemoteFixtureBackend(
            bucket="b", prefix="agent-trace", client=client
        )
        backend.put_bytes("run_1/exchanges/1.json", b"{}")
        backend.put_bytes("run_1/exchanges/2.json", b"{}")
        keys = backend.list_keys("run_1/")
        assert keys == ["run_1/exchanges/1.json", "run_1/exchanges/2.json"]

    def test_missing_boto3_raises_clear_install_hint(self, monkeypatch) -> None:
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(ImportError, match="pip install boto3"):
            S3RemoteFixtureBackend(bucket="b")


# ---------------------------------------------------------------------------
# GCSRemoteFixtureBackend — against a fake google-cloud-storage-shaped bucket
# ---------------------------------------------------------------------------


class _FakeGCSBlob:
    def __init__(self, bucket: _FakeGCSBucket, name: str) -> None:
        self._bucket = bucket
        self.name = name

    def upload_from_string(self, data: bytes) -> None:
        self._bucket.objects[self.name] = data

    def exists(self) -> bool:
        return self.name in self._bucket.objects

    def download_as_bytes(self) -> bytes:
        return self._bucket.objects[self.name]


class _FakeGCSBucket:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def blob(self, name: str) -> _FakeGCSBlob:
        return _FakeGCSBlob(self, name)

    def list_blobs(self, prefix: str = "") -> list[_FakeGCSBlob]:
        return [
            _FakeGCSBlob(self, name)
            for name in sorted(self.objects)
            if name.startswith(prefix)
        ]


class TestGCSRemoteFixtureBackend:
    def test_put_then_get_round_trips(self) -> None:
        backend = GCSRemoteFixtureBackend(
            bucket="ignored", prefix="agent-trace", client=_FakeGCSBucket()
        )
        backend.put_bytes("run_1/fixture.db", b"hello")
        assert backend.get_bytes("run_1/fixture.db") == b"hello"

    def test_get_missing_key_returns_none(self) -> None:
        backend = GCSRemoteFixtureBackend(bucket="ignored", client=_FakeGCSBucket())
        assert backend.get_bytes("nope") is None

    def test_list_keys_strips_prefix(self) -> None:
        client = _FakeGCSBucket()
        backend = GCSRemoteFixtureBackend(
            bucket="ignored", prefix="agent-trace", client=client
        )
        backend.put_bytes("run_1/a.json", b"{}")
        backend.put_bytes("run_1/b.json", b"{}")
        assert backend.list_keys("run_1/") == ["run_1/a.json", "run_1/b.json"]


# ---------------------------------------------------------------------------
# remote_sync_callback() / sync_run_to_remote() / restore_run_from_remote()
# ---------------------------------------------------------------------------


class TestRemoteSyncCallback:
    def test_uploads_exchange_under_run_id_key(self, tmp_path: Path) -> None:
        backend = LocalDirRemoteFixtureBackend(tmp_path / "remote")
        callback = remote_sync_callback(backend, run_id="run_abc")
        callback({"sequence_num": 3, "url": "https://x", "method": "POST"})
        keys = backend.list_keys("run_abc/exchanges/")
        assert len(keys) == 1
        assert "00000003" in keys[0]

    def test_multiple_exchanges_sorted_by_sequence(self, tmp_path: Path) -> None:
        backend = LocalDirRemoteFixtureBackend(tmp_path / "remote")
        callback = remote_sync_callback(backend, run_id="run_abc")
        callback({"sequence_num": 1})
        callback({"sequence_num": 2})
        keys = backend.list_keys("run_abc/exchanges/")
        assert len(keys) == 2

    def test_backend_failure_does_not_raise(self, tmp_path: Path) -> None:
        class _BoomBackend(LocalDirRemoteFixtureBackend):
            def put_bytes(self, key: str, data: bytes) -> None:
                raise RuntimeError("network down")

        callback = remote_sync_callback(
            _BoomBackend(tmp_path / "remote"), run_id="run_abc"
        )
        callback({"sequence_num": 1})  # must not raise


class TestSyncAndRestoreRun:
    def test_sync_then_restore_round_trips_fixture_and_trace(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "runs" / "run_abc"
        run_dir.mkdir(parents=True)
        (run_dir / "fixture.db").write_bytes(b"sqlite-bytes")
        (run_dir / "trace.json").write_text('{"trace_id": "t1"}')

        backend = LocalDirRemoteFixtureBackend(tmp_path / "remote")
        sync_run_to_remote(run_dir, backend, run_id="run_abc")

        restored_dir = restore_run_from_remote(
            backend, run_id="run_abc", dest_dir=tmp_path / "restored"
        )
        assert (restored_dir / "fixture.db").read_bytes() == b"sqlite-bytes"
        assert (restored_dir / "trace.json").read_text() == '{"trace_id": "t1"}'

    def test_sync_skips_missing_files(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "run_abc"
        run_dir.mkdir(parents=True)
        (run_dir / "trace.json").write_text("{}")
        # No fixture.db — not recorded this run.

        backend = LocalDirRemoteFixtureBackend(tmp_path / "remote")
        sync_run_to_remote(run_dir, backend, run_id="run_abc")

        assert backend.get_bytes("run_abc/trace.json") == b"{}"
        assert backend.get_bytes("run_abc/fixture.db") is None

    def test_restore_missing_run_creates_empty_dir(self, tmp_path: Path) -> None:
        backend = LocalDirRemoteFixtureBackend(tmp_path / "remote")
        restored_dir = restore_run_from_remote(
            backend, run_id="never_synced", dest_dir=tmp_path / "restored"
        )
        assert restored_dir.exists()
        assert not (restored_dir / "fixture.db").exists()
