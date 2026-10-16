from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db import get_session_factory
from apps.api.services.storage import ObjectStore, build_object_store


async def get_session() -> AsyncIterator[AsyncSession]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session


@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore:
    return build_object_store()
