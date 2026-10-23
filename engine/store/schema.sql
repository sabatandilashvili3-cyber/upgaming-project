-- One Postgres database holds everything the engine persists.
-- Idempotent: safe to run on every start.
create extension if not exists vector;

create table if not exists documents (
    doc_id        text primary key,            -- "postmortems/INC-2026-016_postmortem.docx", "changelog/x.csv#CHG-3340"
    source        text not null,               -- uploaded file name; rows of one CSV share it
    format        text not null,
    title         text not null,
    text          text not null,
    content_hash  text not null,
    version       int  not null default 1,
    duplicate_of  text,                        -- exact duplicate: not chunked, resolves to this doc
    near_duplicate_of text,                    -- near duplicate: chunked, merged in results
    meta          jsonb not null default '{}',
    created_at    timestamptz not null default now(),
    updated_at    timestamptz not null default now()
);
create index if not exists documents_source on documents (source);
create index if not exists documents_hash on documents (content_hash);

create table if not exists chunks (
    chunk_id   text primary key,
    doc_id     text not null references documents (doc_id) on delete cascade,
    ord        int  not null,
    text       text not null,
    meta       jsonb not null default '{}',
    embedding  vector(384) not null
);
create index if not exists chunks_doc on chunks (doc_id);
-- HNSW: approximate nearest-neighbour graph; cosine distance because vectors are normalised
create index if not exists chunks_embedding on chunks using hnsw (embedding vector_cosine_ops);

-- MinHash LSH bands, so a new upload can be checked for near duplicates without
-- comparing it to every stored document (see engine/ingest/dedup.py)
create table if not exists lsh_bands (
    band    smallint not null,
    value   bigint   not null,       -- hash of one band of the MinHash signature
    doc_id  text     not null references documents (doc_id) on delete cascade
);
create index if not exists lsh_lookup on lsh_bands (band, value);

-- one row per question asked: the full investigation trace, for the UI and for cost tracking
create table if not exists traces (
    trace_id    text primary key,
    question    text not null,
    report      jsonb,
    steps       jsonb not null default '[]',
    model       text,
    tokens_in   int default 0,
    tokens_out  int default 0,
    cost_usd    numeric(10, 6) default 0,
    latency_ms  int,
    error       text,
    created_at  timestamptz not null default now()
);

-- LLM response cache: same prompt + model + params -> same answer, no second bill
create table if not exists llm_cache (
    key        text primary key,
    response   jsonb not null,
    created_at timestamptz not null default now()
);
