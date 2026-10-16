from __future__ import annotations

import os
import re
import shutil
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from apps.api.config import Settings, get_settings


class StorageError(RuntimeError):
    pass


_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w.\-]+", re.UNICODE)


def safe_filename(value: str, *, fallback: str = "upload.bin", max_length: int = 200) -> str:
    """Return a path-component-safe filename while preserving Unicode letters."""
    normalized = unicodedata.normalize("NFKC", str(value)).replace("\\", "/")
    basename = Path(normalized).name.strip().lstrip(".")
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", basename).strip("._")
    return cleaned[:max_length] or fallback


class ObjectStore(ABC):
    @abstractmethod
    def put_stream(self, key: str, stream: BinaryIO, *, content_type: str | None = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def local_path(self, key: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def size(self, key: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def read_range(self, key: str, start: int, end: int) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def exists(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def healthcheck(self) -> None:
        raise NotImplementedError

    def asset_url(self, key: str) -> str:
        return f"/v1/assets/{key}"


def _validate_key(key: str) -> str:
    normalized = key.strip().lstrip("/")
    if not normalized or ".." in Path(normalized).parts:
        raise StorageError("对象键无效")
    return normalized


class LocalObjectStore(ObjectStore):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        normalized = _validate_key(key)
        path = (self.root / normalized).resolve()
        if self.root not in path.parents and path != self.root:
            raise StorageError("对象路径越界")
        return path

    def put_stream(self, key: str, stream: BinaryIO, *, content_type: str | None = None) -> int:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
        size = 0
        try:
            with temporary.open("wb") as output:
                while chunk := stream.read(1024 * 1024):
                    output.write(chunk)
                    size += len(chunk)
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return size

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
    ) -> int:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
        try:
            temporary.write_bytes(data)
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return len(data)

    def local_path(self, key: str) -> Path:
        path = self._path(key)
        if not path.exists():
            raise StorageError(f"对象不存在: {key}")
        return path

    def read_bytes(self, key: str) -> bytes:
        return self.local_path(key).read_bytes()

    def size(self, key: str) -> int:
        return self.local_path(key).stat().st_size

    def read_range(self, key: str, start: int, end: int) -> bytes:
        with self.local_path(key).open("rb") as handle:
            handle.seek(start)
            return handle.read(end - start + 1)

    def delete_prefix(self, prefix: str) -> None:
        normalized = _validate_key(prefix)
        target = self._path(normalized)
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        try:
            return self._path(key).exists()
        except StorageError:
            return False

    def healthcheck(self) -> None:
        if not self.root.is_dir():
            raise StorageError(f"本地存储目录不可用: {self.root}")


class S3ObjectStore(ObjectStore):
    def __init__(self, settings: Settings, cache_root: Path) -> None:
        self.bucket = settings.s3_bucket
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
            config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    def put_stream(self, key: str, stream: BinaryIO, *, content_type: str | None = None) -> int:
        normalized = _validate_key(key)
        size = 0
        path = self.cache_root / normalized
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output:
            while chunk := stream.read(1024 * 1024):
                output.write(chunk)
                size += len(chunk)
        self.client.upload_file(
            str(path),
            self.bucket,
            normalized,
            ExtraArgs={"ContentType": content_type} if content_type else None,
        )
        return size

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
    ) -> int:
        normalized = _validate_key(key)
        kwargs = {"ContentType": content_type} if content_type else {}
        self.client.put_object(Bucket=self.bucket, Key=normalized, Body=data, **kwargs)
        return len(data)

    def local_path(self, key: str) -> Path:
        normalized = _validate_key(key)
        path = self.cache_root / normalized
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, normalized, str(path))
        return path

    def read_bytes(self, key: str) -> bytes:
        return self.local_path(key).read_bytes()

    def size(self, key: str) -> int:
        response = self.client.head_object(Bucket=self.bucket, Key=_validate_key(key))
        return int(response["ContentLength"])

    def read_range(self, key: str, start: int, end: int) -> bytes:
        response = self.client.get_object(
            Bucket=self.bucket,
            Key=_validate_key(key),
            Range=f"bytes={start}-{end}",
        )
        return response["Body"].read()

    def delete_prefix(self, prefix: str) -> None:
        normalized = _validate_key(prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=normalized):
            objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            if objects:
                self.client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": objects},
                )

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=_validate_key(key))
            return True
        except ClientError:
            return False

    def healthcheck(self) -> None:
        self.client.head_bucket(Bucket=self.bucket)


def build_object_store(settings: Settings | None = None) -> ObjectStore:
    current = settings or get_settings()
    if current.storage_backend.lower() == "s3":
        return S3ObjectStore(current, current.storage_root_path / "_s3_cache")
    if current.storage_backend.lower() != "local":
        raise StorageError(f"不支持的对象存储后端: {current.storage_backend}")
    return LocalObjectStore(current.storage_root_path / "objects")


def job_prefix(job_id: str) -> str:
    return f"jobs/{job_id}"


def raw_key(job_id: str, filename: str) -> str:
    suffix = Path(filename).suffix.lower() or ".bin"
    return f"{job_prefix(job_id)}/raw/original{suffix}"


def derived_key(job_id: str, filename: str) -> str:
    return f"{job_prefix(job_id)}/derived/{filename}"
