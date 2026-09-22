-- schema.025.sql -- derived load-bearing (PageRank) authority scores.
--
-- Adds four nullable DOUBLE PRECISION columns to ``inquiries``, one per
-- citation-relation graph the authority sweep ranks: ``proves``, ``favors``,
-- ``cited_by`` (bibliographic ``cites_paper``), and ``issue``
-- (``requires``/``narrows``). NULL means "not yet computed"; the sweep writes a
-- score only on the kinds a relation targets.
--
-- The baseline ``schema.sql`` carries the same columns for a fresh install; a
-- fresh DB records this migration applied WITHOUT executing it, an existing DB
-- records the baseline unrun and executes only this file, so the two must stay
-- in step -- pinned by ``schema_migration_test.py``.
--
-- Numbered 025: the deployed ledger holds through schema.024.sql.
--
-- Purely additive DDL into columns the old build never reads, safe against the
-- OLD code and not downtime (run it against the live database with the old
-- server still serving; see docs/db_schema_migration.md).
ALTER TABLE inquiries ADD COLUMN IF NOT EXISTS proves_authority DOUBLE PRECISION;
ALTER TABLE inquiries ADD COLUMN IF NOT EXISTS favors_authority DOUBLE PRECISION;
ALTER TABLE inquiries ADD COLUMN IF NOT EXISTS cited_by_authority DOUBLE PRECISION;
ALTER TABLE inquiries ADD COLUMN IF NOT EXISTS issue_authority DOUBLE PRECISION;
