"""Creation-time edge guards on the real PGlite and HTTP write paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

from treekle import from_plain, json_loads

import httpx2
import pytest
import pytest_asyncio

from trackinizer.conftest import new_uuid
from trackinizer.lib.postgres.testing import reset_schema
from trackinizer.server.api.app import app
from trackinizer.server.api.canvas_test_support import seed_accounts
from trackinizer.server.api.conftest import make_test_identity
from trackinizer.server.auth import current_user
from trackinizer.server.embedders.stub import StubEmbedder
from trackinizer.server.store.core import Store
from trackinizer.wire.bodies import SubmitBelief, SubmitExperiment, SubmitIssue


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from uuid import UUID

    from trackinizer.lib.postgres import PGliteEngine
    from trackinizer.types.edges import Edge


pytestmark = [pytest.mark.db_pglite, pytest.mark.asyncio(loop_scope="session")]


@pytest_asyncio.fixture(loop_scope="session")
async def store(pglite_engine: PGliteEngine) -> AsyncIterator[Store]:
    """Yield an isolated Store using the shared PGlite engine."""
    await reset_schema(pglite_engine)
    built = Store(pglite_engine, embed=StubEmbedder())
    await built.bootstrap()
    yield built


@pytest_asyncio.fixture(loop_scope="session")
async def client(
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[httpx2.AsyncClient]:
    """Serve the isolated Store as a keyless writer through the whole app."""
    await seed_accounts(store)
    monkeypatch.setattr(app.state, "engine", store.engine, raising=False)
    monkeypatch.setattr(app.state, "store", store, raising=False)
    monkeypatch.setitem(
        app.dependency_overrides,
        current_user,
        lambda: make_test_identity(api_key_id=None),
    )
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as http:
        yield http


@pytest.mark.parametrize("edge_kind", ["narrows", "produced_by", "supersedes"])
async def test_a_new_creation_order_edge_cannot_point_to_a_younger_parent(
    store: Store,
    client: httpx2.AsyncClient,
    edge_kind: Edge.Kind,
) -> None:
    older = await _issue(store, title="older", created="2020-01-01T00:00:00Z")
    younger = await _issue(store, title="younger", created="2021-01-01T00:00:00Z")
    before = await store.list_changes()
    rows_before = [await store.get_inquiry(row_id) for row_id in (older, younger)]

    rejected = await client.post(f"/api/edges/{older}/{edge_kind}/{younger}", json={})

    assert rejected.status_code == 409, rejected.text
    detail = from_plain(json_loads(rejected.content, dict[str, object])["detail"], str)
    assert all(text in detail for text in (str(older), edge_kind, str(younger)))
    assert "--backfill" in detail
    assert (
        await store.get_edge(from_id=older, to_id=younger, edge_kind=edge_kind) is None
    )
    assert await store.list_changes() == before
    assert [
        await store.get_inquiry(row_id) for row_id in (older, younger)
    ] == rows_before


async def test_edge_batches_keep_earlier_successes_and_explain_temporal_failure(
    store: Store,
    client: httpx2.AsyncClient,
) -> None:
    older = await _issue(store, title="older", created="2020-01-01T00:00:00Z")
    younger = await _issue(store, title="younger", created="2021-01-01T00:00:00Z")
    response = await client.post(
        "/api/edges/batch",
        json={
            "items": [
                {"from_id": str(older), "to_id": str(younger), "edge_kind": kind}
                for kind in ("requires", "narrows", "supersedes")
            ],
        },
    )

    assert response.status_code == 200, response.text
    body = json_loads(response.content, dict[str, object])
    assert body["ok"] is False
    items = from_plain(body["items"], list[dict[str, object]])
    assert items[0] == {"ok": True}
    assert items[1]["ok"] is False
    assert items[1]["index"] == 1
    assert "--backfill" in from_plain(items[1]["error"], str)
    assert items[2] == {
        "ok": False,
        "index": 2,
        "error": "skipped after earlier failure",
    }
    assert await store.get_edge(from_id=older, to_id=younger, edge_kind="requires")
    for kind in ("narrows", "supersedes"):
        assert (
            await store.get_edge(from_id=older, to_id=younger, edge_kind=kind) is None
        )
    changes = await store.list_changes()
    assert all(
        change.new is None or change.new.peer_edge_kind != "narrows"
        for change in changes
    )


async def test_create_time_narrows_refusal_rolls_back_the_new_inquiry(
    store: Store,
    client: httpx2.AsyncClient,
) -> None:
    parent = await _issue(store, title="parent", created="2021-01-01T00:00:00Z")
    key = new_uuid()
    before = await store.list_changes()
    async with store.engine.acquire() as conn:
        embedding_ids = await conn.fetch(
            "SELECT inquiry_id, model FROM inquiry_embeddings ORDER BY inquiry_id, model",
        )
    response = await client.post(
        "/api/inquiries/issue",
        json={
            "title": "incorrectly older child",
            "created": "2020-01-01T00:00:00Z",
            "idempotency_key": str(key),
            "narrows": [[str(parent), None]],
        },
    )

    assert response.status_code == 409, response.text
    assert "--backfill" in response.text
    assert await store.get_change(key) is None
    assert await store.list_changes() == before
    async with store.engine.acquire() as conn:
        assert [row["id"] for row in await conn.fetch("SELECT id FROM inquiries")] == [
            parent,
        ]
        assert await conn.fetchval("SELECT count(*) FROM edges") == 0
        assert (
            await conn.fetch(
                "SELECT inquiry_id, model FROM inquiry_embeddings ORDER BY inquiry_id, model",
            )
            == embedding_ids
        )


async def test_inquiry_batch_temporal_failure_rolls_back_every_item_and_audit(
    store: Store,
    client: httpx2.AsyncClient,
) -> None:
    keys = [new_uuid(), new_uuid()]
    response = await client.post(
        "/api/inquiries/batch",
        json={
            "items": [
                {
                    "kind": "Issue",
                    "title": title,
                    "created": created,
                    "idempotency_key": str(key),
                }
                for key, title, created in zip(
                    keys,
                    ("older child", "younger parent"),
                    ("2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"),
                    strict=True,
                )
            ],
            "edges": [{"from_index": 0, "to_index": 1, "edge_kind": "narrows"}],
        },
    )

    assert response.status_code == 409, response.text
    assert "--backfill" in response.text
    for key in keys:
        assert await store.get_change(key) is None
    assert await store.list_changes() == []
    async with store.engine.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM inquiries") == 0
        assert await conn.fetchval("SELECT count(*) FROM edges") == 0
        assert await conn.fetchval("SELECT count(*) FROM inquiry_embeddings") == 0


@pytest.mark.parametrize("edge_kind", ["narrows", "produced_by", "supersedes"])
async def test_imported_equal_timestamps_allow_creation_order_edges(
    store: Store,
    client: httpx2.AsyncClient,
    edge_kind: Edge.Kind,
) -> None:
    created = "2020-01-01T00:00:00Z"
    first = await _issue(store, title="first", created=created)
    second = await _issue(store, title="second", created=created)

    response = await client.post(f"/api/edges/{first}/{edge_kind}/{second}", json={})

    assert response.status_code == 200, response.text
    assert await store.get_edge(from_id=first, to_id=second, edge_kind=edge_kind)


@pytest.mark.parametrize("edge_kind", ["proves", "favors"])
async def test_backfilled_experiment_can_cite_a_newer_belief_without_provenance(
    store: Store,
    client: httpx2.AsyncClient,
    edge_kind: Edge.Kind,
) -> None:
    evidence = await store.submit_experiment(
        SubmitExperiment.model_validate(
            {
                "account": "tester@example.com",
                "title": "historical experiment",
                "created": "2020-01-01T00:00:00Z",
            },
        ),
    )
    claim = await store.submit_belief(
        SubmitBelief.model_validate(
            {
                "account": "tester@example.com",
                "title": "later claim",
                "created": "2021-01-01T00:00:00Z",
            },
        ),
    )

    response = await client.post(
        f"/api/edges/{evidence}/{edge_kind}/{claim}",
        json={"valence": 0.7},
    )

    assert response.status_code == 200, response.text
    edge = await store.get_edge(from_id=evidence, to_id=claim, edge_kind=edge_kind)
    assert edge is not None
    assert edge.valence == 0.7
    for row_id in (evidence, claim):
        row = await store.get_inquiry(row_id)
        assert row is not None
        assert row.produced_by == ()
        assert row.produces == ()


@pytest.mark.parametrize("edge_kind", ["narrows", "produced_by", "supersedes"])
async def test_legacy_edges_allow_noop_upsert_annotation_and_removal(
    store: Store,
    client: httpx2.AsyncClient,
    edge_kind: Edge.Kind,
) -> None:
    older = await _issue(store, title="older", created="2020-01-01T00:00:00Z")
    younger = await _issue(store, title="younger", created="2021-01-01T00:00:00Z")
    async with store.engine.acquire() as conn:
        await conn.execute(
            "INSERT INTO edges (from_id, from_kind, to_id, to_kind, edge_kind) "
            "VALUES ($1, 'Issue', $2, 'Issue', $3)",
            older,
            younger,
            edge_kind,
        )
    before = await store.list_changes()
    path = f"/api/edges/{older}/{edge_kind}/{younger}"

    noop = await client.post(path, json={})
    assert noop.status_code == 200, noop.text
    assert json_loads(noop.content, dict[str, object]) == {
        "change_id": None,
        "created": False,
    }
    assert await store.list_changes() == before
    upsert = await client.post(path, json={"note": "legacy context"})
    assert upsert.status_code == 200, upsert.text
    assert json_loads(upsert.content, dict[str, object])["created"] is False
    edge = await store.get_edge(from_id=older, to_id=younger, edge_kind=edge_kind)
    assert edge is not None
    assert edge.note == "legacy context"
    annotation = await client.put(path + "/note", json={"value": "revised context"})
    assert annotation.status_code == 200, annotation.text
    deleted = await client.request("DELETE", path, json={})
    assert deleted.status_code == 200, deleted.text
    assert (
        await store.get_edge(from_id=older, to_id=younger, edge_kind=edge_kind) is None
    )


async def _issue(store: Store, *, title: str, created: str) -> UUID:
    return await store.submit_issue(
        SubmitIssue.model_validate(
            {"account": "tester@example.com", "title": title, "created": created},
        ),
    )


if __name__ == "__main__":
    from trackinizer.lib.testing.runner import test_main

    test_main(__file__)
