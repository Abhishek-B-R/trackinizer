"""Place a ``trax run`` session in a workstream and a station.

A placed session has exactly one ``produced_by`` edge to a workstream Issue, one
``station:<workspace uuid>`` label and, on a git branch, one ``git:<origin>@<branch>``
label. A station holds many sessions; one made for a session also remembers it as
``StationData.session_id``, which never changes who Chat talks to.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit, urlunsplit

import uuid

from treekle import from_plain

from trackinizer.server.notify import notify_after_commit, tx
from trackinizer.server.primitives import validate_edge_creation_order
from trackinizer.server.visuals.workspace_store import (
    new_session_station,
    set_station_workstream,
)
from trackinizer.server.visuals.workstreams import check_workstream, open_scratch
from trackinizer.types.errors import NotFoundError
from trackinizer.wire.wire_science_chat import (
    CHAT_HELPER_CLI,
    CHAT_SESSION_PREFIX,
)
from trackinizer.wire.wire_service_accounts import is_service_address
from trackinizer.wire.wire_sessions import (
    GIT_LABEL_PREFIX,
    STATION_LABEL_PREFIX,
    GitRef,
    PlaceSession,
    SessionPlacement,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection

    from trackinizer.lib.postgres import Conn, DatabaseEngine
    from trackinizer.server.config import Assistant
    from trackinizer.server.inbound import InboundQueue
    from trackinizer.server.store.core import Store
    from trackinizer.server.visuals.catalog import VisualCatalogBody
    from trackinizer.server.visuals.workspaces import WorkspaceState
    from trackinizer.types.inquiries import AgentSession


type AuthorizePlacement = Callable[[Conn, Collection[uuid.UUID]], Awaitable[None]]


_WORKSTREAM_EDGES: Final = (
    "SELECT e.to_id FROM edges AS e JOIN inquiries AS issue ON issue.id = e.to_id "
    "WHERE e.from_id = $1 AND e.edge_kind = 'produced_by' AND e.to_kind = 'Issue' "
    "AND issue.labels @> ARRAY['workstream']::text[]"
)


def git_label(git: GitRef) -> str:
    """Return the ``git:`` label of a checkout, its origin URL without credentials.

    Args:
      git: The origin URL and branch the client sent.

    Returns:
      label: ``git:<origin>@<branch>``.

    """
    origin = git.origin.strip()
    parts = urlsplit(origin)
    if parts.scheme and "@" in parts.netloc:
        origin = urlunsplit(parts._replace(netloc=parts.netloc.rpartition("@")[2]))
    return f"{GIT_LABEL_PREFIX}{origin}@{git.branch.strip()}"


def is_placed(session: AgentSession) -> bool:
    """Say whether a session is one a run places: a person's own ``trax run``.

    Args:
      session: The session.

    Returns:
      placed: False for a service account, a science chat and a ``trax helper``.

    """
    return not (
        is_service_address(session.account)
        or session.cli == CHAT_HELPER_CLI
        or (session.cli_session_id or "").startswith(CHAT_SESSION_PREFIX)
    )


async def place_session(
    engine: DatabaseEngine,
    *,
    store: Store,
    session: AgentSession,
    user_id: uuid.UUID,
    email: str,
    api_key_id: uuid.UUID | None,
    body: PlaceSession,
    catalog: VisualCatalogBody,
    authorize: AuthorizePlacement,
) -> SessionPlacement:
    """Place ``session`` by the first rule that matches, and record it.

    1. ``body.workspace`` is the caller's station: that station and its workstream.
    2. The session is already placed, or has a station made for it: keep them.
    3. The caller's newest earlier session on the same git branch has a workstream:
       that workstream, in a new station.
    4. The caller's Scratch workstream, in a new station.

    The account lock serializes placement with session and station moves. The
    station, provenance edge, and labels commit together, so a retry converges.

    Args:
      engine: Database connection source.
      store: Store that writes the labels and the edge.
      session: The session, opened by the caller's own key.
      user_id: The caller.
      email: The caller's account.
      api_key_id: The key that opened the session.
      body: The station the run started from and its checkout.
      catalog: Trusted visual definitions for a new station.
      authorize: The caller's locked-inquiry guard, applied before edge or label writes.

    Returns:
      placement: The workstream and station; both ``None`` for a session not placed.

    """
    if not is_placed(session):
        return SessionPlacement()
    label = None if body.git is None else git_label(body.git)
    async with notify_after_commit(), engine.acquire() as conn, tx(conn):
        _ = await _lock_account(conn, email=email)
        scratch = await open_scratch(
            engine,
            store=store,
            user_id=user_id,
            email=email,
            conn=conn,
        )
        labels = await conn.fetchval(
            "SELECT labels FROM inquiries WHERE id = $1 FOR UPDATE",
            session.id,
        )
        session = replace(
            session,
            labels=tuple(from_plain(labels, list[str], default=[])) or None,
        )
        workstream_id, workspace_id = await _decide(
            conn,
            session=session,
            user_id=user_id,
            email=email,
            workspace=body.workspace,
            git=label,
            scratch=scratch,
        )
        await validate_edge_creation_order(
            conn,
            from_id=session.id,
            to_id=workstream_id,
            edge_kind="produced_by",
        )
        if workspace_id is None:
            workspace_id = await new_session_station(
                conn,
                user_id=user_id,
                name=f"{session.owner or email} {session.cli or 'session'}",
                catalog=catalog,
                workstream_id=workstream_id,
                session_id=session.id,
            )
        await _record(
            store,
            conn=conn,
            authorize=authorize,
            current=tuple(session.labels or ()),
            session=session,
            api_key_id=api_key_id,
            email=email,
            workstream_id=workstream_id,
            wanted=[
                f"{STATION_LABEL_PREFIX}{workspace_id}",
                *([] if label is None else [label]),
            ],
        )
    return SessionPlacement(workstream_id=workstream_id, workspace_id=workspace_id)


async def move_session(
    engine: DatabaseEngine,
    *,
    store: Store,
    session: AgentSession,
    email: str,
    api_key_id: uuid.UUID | None,
    workstream_id: uuid.UUID,
    authorize: AuthorizePlacement,
) -> bool:
    """Move one session to an owned workstream, and its made-for station if it is alone.

    Args:
      engine: Database connection source.
      store: Store that writes the edge.
      session: The session to move.
      email: The caller's account, which owns the workstream.
      api_key_id: The calling key, if any.
      workstream_id: The destination.
      authorize: The caller's locked-inquiry guard for every touched endpoint.

    Returns:
      moved: False when the destination is not the caller's workstream.

    """
    async with notify_after_commit(), engine.acquire() as conn, tx(conn):
        user_id = await _lock_account(conn, email=email)
        if not await check_workstream(
            engine,
            email=email,
            workstream_id=workstream_id,
            conn=conn,
        ):
            return False
        await _set_workstream(
            store,
            conn=conn,
            authorize=authorize,
            session_id=session.id,
            workstream_id=workstream_id,
            api_key_id=api_key_id,
            actor=email,
        )
        # The station made for it moves only if no other session sits on it.
        await conn.execute(
            "UPDATE visual_workspaces AS w SET station = (coalesce(w.station, "
            "'{}'::jsonb) - 'suggestion') || jsonb_build_object('workstream_id', "
            "$2::text), modified_at = clock_timestamp() "
            "WHERE w.station->>'session_id' = $3 AND w.user_id = $5 AND NOT EXISTS ("
            "SELECT 1 FROM inquiries AS other WHERE other.kind = 'AgentSession' "
            "AND other.id <> $1 AND other.labels @> ARRAY[$4 || w.id::text])",
            session.id,
            str(workstream_id),
            str(session.id),
            STATION_LABEL_PREFIX,
            user_id,
        )
    return True


async def move_station(
    engine: DatabaseEngine,
    *,
    store: Store,
    user_id: uuid.UUID,
    workspace_id: uuid.UUID,
    email: str,
    api_key_id: uuid.UUID | None,
    workstream_id: uuid.UUID,
    inbound: InboundQueue,
    assistant: Assistant | None,
    authorize: AuthorizePlacement,
) -> WorkspaceState | None:
    """Move an owned station and all its session provenance in one transaction.

    Args:
      engine: Database connection source.
      store: Store that writes audited session edges.
      user_id: The station owner.
      workspace_id: The station to move.
      email: The caller's account; only its sessions move.
      api_key_id: The calling key, if any.
      workstream_id: The station's new owned workstream.
      inbound: Session queues used to render the station's partner.
      assistant: The configured assistant, if any.
      authorize: The caller's locked-inquiry guard for every touched endpoint.

    Returns:
      state: The station after its complete move, or None when either id is unavailable.

    """
    async with notify_after_commit(), engine.acquire() as conn, tx(conn):
        _ = await _lock_account(conn, email=email)
        state = await set_station_workstream(
            engine,
            user_id=user_id,
            email=email,
            workspace_id=workspace_id,
            workstream_id=workstream_id,
            inbound=inbound,
            assistant=assistant,
            conn=conn,
        )
        if state is None:
            return None
        rows = await conn.fetch(
            "SELECT id FROM inquiries WHERE kind = 'AgentSession' AND account = $1 "
            "AND $2 = ANY(labels) ORDER BY created DESC, id",
            email,
            f"{STATION_LABEL_PREFIX}{workspace_id}",
        )
        for row in rows:
            await _set_workstream(
                store,
                conn=conn,
                authorize=authorize,
                session_id=from_plain(row["id"], uuid.UUID),
                workstream_id=workstream_id,
                api_key_id=api_key_id,
                actor=email,
            )
    return state


async def _decide(
    conn: Conn,
    *,
    session: AgentSession,
    user_id: uuid.UUID,
    email: str,
    workspace: uuid.UUID | None,
    git: str | None,
    scratch: uuid.UUID,
) -> tuple[uuid.UUID, uuid.UUID | None]:
    """Apply the four rules; the station is ``None`` when a new one is to be made."""
    if workspace is not None:
        station = await conn.fetchrow(
            "SELECT station->>'workstream_id' AS workstream FROM visual_workspaces "
            "WHERE id = $1 AND user_id = $2",
            workspace,
            user_id,
        )
        if station is not None:
            owned = await _owned(conn, email=email, workstream=station["workstream"])
            return owned or scratch, workspace
    edge = await conn.fetchval(_WORKSTREAM_EDGES, session.id)
    made = await conn.fetchval(
        "SELECT id FROM visual_workspaces WHERE user_id = $1 "
        "AND (id::text = $3 OR station->>'session_id' = $2) ORDER BY created_at, id LIMIT 1",
        user_id,
        str(session.id),
        next(
            (
                label.removeprefix(STATION_LABEL_PREFIX)
                for label in session.labels or ()
                if label.startswith(STATION_LABEL_PREFIX)
            ),
            "-",
        ),
    )
    if edge is not None or made is not None:
        workstream = from_plain(edge, uuid.UUID) if edge is not None else scratch
        return workstream, None if made is None else from_plain(made, uuid.UUID)
    if git is not None:
        earlier = await conn.fetchval(
            "SELECT e.to_id FROM inquiries AS past JOIN edges AS e "
            "ON e.from_id = past.id AND e.edge_kind = 'produced_by' "
            "AND e.to_kind = 'Issue' JOIN inquiries AS issue ON issue.id = e.to_id "
            "WHERE past.kind = 'AgentSession' AND past.account = $1 "
            "AND $2 = ANY(past.labels) AND past.id <> $3 AND issue.owner = $1 "
            "AND issue.labels @> ARRAY['workstream']::text[] "
            "ORDER BY past.created DESC, past.id DESC LIMIT 1",
            email,
            git,
            session.id,
        )
        if earlier is not None:
            return from_plain(earlier, uuid.UUID), None
    return scratch, None


async def _owned(
    conn: Conn,
    *,
    email: str,
    workstream: object,
) -> uuid.UUID | None:
    """Return a station's stored workstream id if it is an owned workstream Issue."""
    if workstream is None:
        return None
    found = await conn.fetchval(
        "SELECT id FROM inquiries WHERE id = $1::uuid AND kind = 'Issue' "
        "AND owner = $2 AND labels @> ARRAY['workstream']::text[]",
        from_plain(workstream, str),
        email,
    )
    return None if found is None else from_plain(found, uuid.UUID)


async def _record(
    store: Store,
    *,
    conn: Conn,
    authorize: AuthorizePlacement,
    current: tuple[str, ...],
    session: AgentSession,
    api_key_id: uuid.UUID | None,
    email: str,
    workstream_id: uuid.UUID,
    wanted: list[str],
) -> None:
    """Make the session's edge and its station and git labels exactly the wanted."""
    await _set_workstream(
        store,
        conn=conn,
        authorize=authorize,
        session_id=session.id,
        workstream_id=workstream_id,
        api_key_id=api_key_id,
        actor=email,
    )
    families = (STATION_LABEL_PREFIX, GIT_LABEL_PREFIX)
    for old in current:
        if old.startswith(families) and old not in wanted:
            _ = await store.remove_label(
                session.id,
                old,
                api_key_id=api_key_id,
                actor=email,
                conn=conn,
            )
    for label in wanted:
        if label not in current:
            _ = await store.add_label(
                session.id,
                label,
                api_key_id=api_key_id,
                actor=email,
                conn=conn,
            )


async def _set_workstream(
    store: Store,
    *,
    conn: Conn,
    authorize: AuthorizePlacement,
    session_id: uuid.UUID,
    workstream_id: uuid.UUID,
    api_key_id: uuid.UUID | None,
    actor: str,
) -> None:
    """Leave one ``produced_by`` edge from the session to a workstream: the given."""
    rows = await conn.fetch(_WORKSTREAM_EDGES, session_id)
    await authorize(
        conn,
        {
            session_id,
            workstream_id,
            *(from_plain(row["to_id"], uuid.UUID) for row in rows),
        },
    )
    await validate_edge_creation_order(
        conn,
        from_id=session_id,
        to_id=workstream_id,
        edge_kind="produced_by",
    )
    for row in rows:
        old = from_plain(row["to_id"], uuid.UUID)
        if old != workstream_id:
            _ = await store.remove_edge(
                from_id=session_id,
                to_id=old,
                edge_kind="produced_by",
                api_key_id=api_key_id,
                actor=actor,
                conn=conn,
            )
    _ = await store.add_edge(
        from_id=session_id,
        to_id=workstream_id,
        edge_kind="produced_by",
        api_key_id=api_key_id,
        actor=actor,
        conn=conn,
    )


async def _lock_account(conn: Conn, *, email: str) -> uuid.UUID:
    """Serialize placement and moves on the account's durable owner row."""
    user_id = await conn.fetchval(
        "SELECT id FROM users WHERE email = $1 FOR UPDATE",
        email,
    )
    if user_id is None:
        raise NotFoundError("User not found")
    return from_plain(user_id, uuid.UUID)
