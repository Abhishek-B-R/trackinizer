"""Keep session placement and station moves atomic under concurrent requests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import asyncio
import uuid

from treekle import from_plain, json_loads

import httpx2
import pytest
import pytest_asyncio

from trackinizer.server.api.app import app
from trackinizer.server.api.canvas_test_support import (
    ASSISTANT_CONFIG,
    USER_AGENT,
    browser,
    seed_accounts,
)
from trackinizer.server.api.conftest import (
    _serving,
    install_identity,
    make_test_identity,
)
from trackinizer.server.chat_hub import ChatHub
from trackinizer.server.inbound import InboundQueue
from trackinizer.server.notify import tx
from trackinizer.wire.bodies import SubmitAgentSession, SubmitIssue


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from asyncpg import Record

    from trackinizer.server.auth import Role
    from trackinizer.server.store.core import Store


type _Client = tuple[httpx2.AsyncClient, Store]
type _Snapshot = tuple[list[Record], list[Record], list[Record], list[Record]]


@pytest.fixture(autouse=True)
def served(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use isolated app configuration and session queues."""
    monkeypatch.setattr(app.state, "config", ASSISTANT_CONFIG, raising=False)
    monkeypatch.setattr(app.state, "inbound", InboundQueue(), raising=False)
    monkeypatch.setattr(app.state, "hub", ChatHub(), raising=False)


@pytest_asyncio.fixture(loop_scope="session")
async def postgres_route_client(integ_store: Store) -> AsyncIterator[_Client]:
    """Serve the app on a real connection pool for concurrent lock checks."""
    with _serving(
        integ_store.engine,
        integ_store,
        make_test_identity(api_key_id=None),
    ):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as client:
            yield client, integ_store


async def _scenario(
    client: httpx2.AsyncClient,
    store: Store,
    *,
    session_year: int = 2021,
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID, uuid.UUID]:
    """Make two workstreams, an old station, and an unplaced session."""
    await seed_accounts(store)
    workstreams = [
        await store.submit_issue(
            SubmitIssue(
                title=f"Workstream {year}",
                account=USER_AGENT.email,
                owner=USER_AGENT.email,
                labels=["workstream", "scratch"] if year == 2019 else ["workstream"],
                created=datetime(year, 1, 1, tzinfo=UTC),
            ),
            actor=USER_AGENT.email,
        )
        for year in (2019, 2021)
    ]
    old, target = workstreams
    session = await store.submit_agentsession(
        SubmitAgentSession(
            title="Imported session",
            account=USER_AGENT.email,
            cli="codex",
            created=datetime(session_year, 1, 1, tzinfo=UTC),
        ),
        audience="open",
        actor=USER_AGENT.email,
    )
    browser()
    made = await client.post("/api/workspaces/new", json={"workstream_id": str(old)})
    assert made.status_code == 200, made.text
    station = uuid.UUID(
        from_plain(json_loads(made.content, dict[str, object])["id"], str),
    )
    return old, target, station, session


async def _snapshot(store: Store) -> _Snapshot:
    """Capture complete placement, inquiry, edge, and audit identities."""
    async with store.engine.acquire() as conn:
        return (
            list(await conn.fetch("SELECT * FROM visual_workspaces ORDER BY id")),
            list(await conn.fetch("SELECT * FROM inquiries ORDER BY id")),
            list(
                await conn.fetch(
                    "SELECT * FROM edges ORDER BY from_id, to_id, edge_kind",
                ),
            ),
            list(await conn.fetch("SELECT * FROM change_log ORDER BY id")),
        )


async def _move(
    client: httpx2.AsyncClient,
    *,
    kind: str,
    station: uuid.UUID,
    session: uuid.UUID,
    target: uuid.UUID,
) -> httpx2.Response:
    """Request one specialized placement mutation as the installed principal."""
    if kind == "station":
        return await client.put(
            f"/api/workspaces/{station}/workstream",
            json={"workstream_id": str(target)},
        )
    if kind == "session":
        return await client.put(
            f"/api/sessions/{session}/workstream",
            json={"workstream_id": str(target)},
        )
    return await client.post(
        f"/api/sessions/{session}/place",
        json={"workspace": str(station)},
    )


@pytest.mark.db_postgres
@pytest.mark.asyncio(loop_scope="session")
async def test_station_moves_and_placement_serialize_on_the_account_lock(
    postgres_route_client: _Client,
) -> None:
    """A waiting historical placement cannot join a station midway through its move."""
    client, store = postgres_route_client
    _old, target, station, session = await _scenario(client, store, session_year=2020)
    tasks: list[asyncio.Task[httpx2.Response]] = []
    try:
        async with store.engine.acquire() as conn, tx(conn):
            await conn.execute(
                "SELECT id FROM users WHERE email = $1 FOR UPDATE",
                USER_AGENT.email,
            )
            tasks.append(
                asyncio.create_task(
                    _move(
                        client,
                        kind="station",
                        station=station,
                        session=session,
                        target=target,
                    ),
                ),
            )
            finished, _pending = await asyncio.wait(tasks, timeout=0.5)
            assert finished == set(), "Station move bypassed the account lock"
            tasks.append(
                asyncio.create_task(
                    _move(
                        client,
                        kind="place",
                        station=station,
                        session=session,
                        target=target,
                    ),
                ),
            )
            finished, _pending = await asyncio.wait(tasks, timeout=0.05)
            assert finished == set(), "Placement bypassed the account lock"
        moved, placed = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert (moved.status_code, placed.status_code) == (200, 409)
    async with store.engine.acquire() as conn:
        assert await conn.fetchval(
            "SELECT station->>'workstream_id' FROM visual_workspaces WHERE id = $1",
            station,
        ) == str(target)
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM edges WHERE from_id = $1",
                session,
            )
            == 0
        )
        assert (
            await conn.fetchval("SELECT labels FROM inquiries WHERE id = $1", session)
            is None
        )


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("kind", ["station", "session", "place"])
async def test_cycle_rejected_placement_rolls_back_all_prior_writes(
    pglite_route_client: _Client,
    kind: str,
) -> None:
    """Late cycle failure preserves complete station, labels, edges, and audit rows."""
    client, store = pglite_route_client
    _old, target, station, session = await _scenario(client, store)
    placed = await _move(
        client,
        kind="place",
        station=station,
        session=session,
        target=target,
    )
    assert placed.status_code == 200, placed.text
    _ = await store.add_edge(
        from_id=target,
        to_id=session,
        edge_kind="produced_by",
        actor=USER_AGENT.email,
    )
    if kind == "place":
        made = await client.post(
            "/api/workspaces/new",
            json={"workstream_id": str(target)},
        )
        assert made.status_code == 200, made.text
        station = uuid.UUID(
            from_plain(json_loads(made.content, dict[str, object])["id"], str),
        )
    before = await _snapshot(store)

    refused = await _move(
        client,
        kind=kind,
        station=station,
        session=session,
        target=target,
    )

    assert refused.status_code == 409, refused.text
    assert await _snapshot(store) == before


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_rejected_first_placement_rolls_back_lazy_scratch_creation(
    pglite_route_client: _Client,
) -> None:
    """A lazily created Scratch and its audit share the placement's rollback."""
    client, store = pglite_route_client
    old, target, station, session = await _scenario(client, store)
    _ = await store.remove_label(old, "scratch", actor=USER_AGENT.email)
    async with store.engine.acquire() as conn:
        await conn.execute("UPDATE inquiries SET locked = TRUE WHERE id = $1", old)
    before = await _snapshot(store)

    refused = await _move(
        client,
        kind="place",
        station=station,
        session=session,
        target=target,
    )

    assert refused.status_code == 403, refused.text
    assert await _snapshot(store) == before


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("kind", ["station", "session", "place"])
@pytest.mark.parametrize("locked_endpoint", ["session", "old", "target"])
@pytest.mark.parametrize("role", ["writer", "admin"])
async def test_specialized_placement_obeys_the_locked_inquiry_contract(
    pglite_route_client: _Client,
    kind: str,
    locked_endpoint: str,
    role: Role,
) -> None:
    """Every touched edge endpoint uses the generic route's admin-only lock rule."""
    client, store = pglite_route_client
    old, target, station, session = await _scenario(client, store)
    placed = await _move(
        client,
        kind="place",
        station=station,
        session=session,
        target=target,
    )
    assert placed.status_code == 200, placed.text
    if kind == "place":
        made = await client.post(
            "/api/workspaces/new",
            json={"workstream_id": str(target)},
        )
        assert made.status_code == 200, made.text
        station = uuid.UUID(
            from_plain(json_loads(made.content, dict[str, object])["id"], str),
        )
    async with store.engine.acquire() as conn:
        await conn.execute(
            "UPDATE inquiries SET locked = TRUE WHERE id = $1",
            {"session": session, "old": old, "target": target}[locked_endpoint],
        )
    install_identity(make_test_identity(api_key_id=None, role=role))
    before = await _snapshot(store)

    response = await _move(
        client,
        kind=kind,
        station=station,
        session=session,
        target=target,
    )

    assert response.status_code == (
        403 if role == "writer" else 204 if kind == "session" else 200
    ), response.text
    if role == "writer":
        assert await _snapshot(store) == before


if __name__ == "__main__":
    from trackinizer.lib.testing.runner import test_main

    test_main(__file__)
