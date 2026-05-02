# @mem0/bun

Production-grade local-first TypeScript rewrite of Mem0 OSS, built for [Bun](https://bun.com).

This package is a **full** Mem0 v3 (mem0+) reimplementation:

- LLM-driven additive extraction (single-call ADD/UPDATE/DELETE/NONE flow with attribution).
- Hybrid retrieval: cosine + BM25 keyword + entity-store boost + graph-traversal boost.
- **Local-first persistence by default** — every store runs on `bun:sqlite`:
  - `SqliteVectorStore` with FTS5 keyword search (built-in BM25) and JSON-encoded vectors.
  - `SqliteGraphStore` for triples + embedded entity vectors.
  - `SqliteHistoryManager` for memory change history.
  - All single-file, no native extensions, no external services.
- **Pluggable abstract interfaces** — `VectorStore` / `GraphStore` / `Embedder` / `LLM` / `HistoryManager` are drop-in extension points so you can swap in Qdrant/Neo4j/Kuzu/etc. without touching `Memory`.
- Graph memory: subject-relation-object triples with multi-hop traversal.
- Procedural memory: compress agent execution traces into a single structured summary.
- Multi-language NER: enhanced regex (CJK / email / URL / phone / acronym / code identifier) + optional LLM and Hybrid extractors.
- Retries with exponential backoff for LLM and embedder calls.
- Provider integrations (OpenAI, Anthropic, Ollama, Qdrant) over `fetch` — no SDK dependencies.
- **100% line + function coverage** under `bun test`.

## Install

```bash
cd mem0-bun
bun install
```

## Quick start

```ts
import { Memory } from "@mem0/bun";

const memory = await Memory.create({
  llm: { provider: "openai", config: { apiKey: process.env.OPENAI_API_KEY } },
  embedder: { provider: "openai", config: { apiKey: process.env.OPENAI_API_KEY } },
  // Persist to disk by passing a `path`. Omit it for `:memory:` mode.
  vectorStore: {
    provider: "sqlite",
    config: { path: "./mem0.vec.db", dimension: 1536 },
  },
  historyStore: { provider: "sqlite", config: { path: "./mem0.history.db" } },
  graphStore:   { provider: "sqlite", config: { path: "./mem0.graph.db" } },
  enableGraph: true,                     // turns on mem0+ graph memory
  entityExtractor: { mode: "hybrid" },   // local NER → LLM fallback for non-Latin
  retry: { attempts: 3, baseDelayMs: 200 },
  scoringWeights: { semantic: 1.0, bm25: 0.5, entity: 0.3, graph: 0.4 },
});

await memory.add(
  [
    { role: "user", content: "Hi, my name is Alice and I live in Paris. I love hiking." },
  ],
  { userId: "alice" },
);

const search = await memory.search("Where does Alice live?", {
  filters: { user_id: "alice" },
});
console.log(search.results);
```

## Architecture

```
src/
├── memory/memory.ts                V3 phased pipeline (extract → diff → batch persist → entity link → graph link)
├── llms/                           base + openai + anthropic + ollama + mock (fetch-based, retry-wrapped)
├── embeddings/                     base + openai + ollama + mock
├── vector_stores/
│   ├── base.ts                     abstract — extension point for any backend
│   ├── sqlite.ts                   default — bun:sqlite + FTS5 keyword search + JSON vectors
│   ├── memory.ts                   in-process: cosine + BM25 (in-memory inverted index)
│   └── qdrant.ts                   REST-API based; no full-text
├── graphs/
│   ├── base.ts                     GraphStore abstract — addTriples / searchByEntities / deleteByMemoryId
│   ├── sqlite.ts                   default — bun:sqlite triples table + indexed entity columns
│   └── in_memory.ts                in-process variant
├── entities/
│   ├── base.ts                     EntityExtractor interface
│   ├── local.ts                    compromise + regex augmentations (default)
│   ├── llm.ts                      LLM-driven NER (multi-language)
│   └── hybrid.ts                   local first, LLM fallback for non-Latin / empty-result text
├── storage/                        bun:sqlite + in-memory history managers
├── prompts/                        FACT_RETRIEVAL, UPDATE_MEMORY, ADDITIVE_EXTRACTION, TRIPLE_EXTRACTION, PROCEDURAL
├── config/factory.ts               provider factories + resolveConfig (zod)
├── utils/
│   ├── lemmatization.ts            Porter stemmer (via natural) + tokenizer + curated stopwords
│   ├── entity_extraction.ts        compromise-based NER (person / place / org / date / value / topic)
│   ├── bm25.ts                     BM25 + sigmoid normalization + per-query parameter tuning
│   ├── scoring.ts                  semantic + BM25 + entity + graph score fusion + ranking
│   └── retry.ts                    exponential-backoff retry helper
└── types/                          zod schemas for the entire config surface
```

## Memory.add (V3 phased pipeline)

1. Normalize messages, validate filters.
2. Embed transcript, semantic-search top 10 existing memories scoped by filters.
3. **Single LLM call** — `ADDITIVE_EXTRACTION_PROMPT` produces `{ memory: [{ id, text, event, old_memory?, attributed_to? }] }`. Agent-only sessions (no `user_id`) get `AGENT_CONTEXT_SUFFIX` appended for assistant-persona bias.
4. Hash-dedup ADD/UPDATE entries, batch-embed their texts.
5. Apply ADD / UPDATE / DELETE / NONE actions; per-record errors don't abort the batch (they surface as `metadata.event = "ERROR"`).
6. **Entity-store linking** — extract entities (compromise NER), batch-embed unique ones, search the sibling collection for matches at sim ≥ 0.95 → update linkedMemoryIds, else insert new.
7. **Graph triples** (when `enableGraph: true`) — single LLM call extracts triples; each triple is linked to every newly-affected memory id and embedded for fuzzy entity matching.
8. Append raw messages to history for context in future turns.

## Memory.search (hybrid retrieval)

1. Lemmatize query (Porter, stopwords stripped); extract query entities.
2. Embed the query.
3. Over-fetch semantic candidates (`max(topK × 4, 60)`).
4. BM25 keyword search over `payload.textLemmatized`, then sigmoid-normalize scores using per-query midpoint/steepness.
5. Entity-store boost: search the sibling collection by query entities, apply spread-attenuated boost (`sim × weight × 1/(1 + 0.001(n-1)²)`).
6. Graph subgraph boost (when graph enabled): seed with query entities, traverse `hops=2` with hop-attenuation, boost the linked memory ids.
7. Score fusion: `final = w_sem·sem + w_bm25·bm25 + w_ent·ent + w_graph·graph` with configurable weights.
8. Threshold + topK truncation.

## Memory.addProcedural (agent traces)

```ts
await memory.addProcedural(
  [
    { role: "agent", content: "Open URL https://example.com" },
    { role: "tool",  content: "200 OK\n<html>...</html>" },
    { role: "agent", content: "Extracted blog titles" },
  ],
  { agentId: "agent-1", metadata: { task: "scrape blog" } },
);
```

Uses `PROCEDURAL_MEMORY_PROMPT` (verbatim from upstream mem0) to preserve every action's exact output, then stores the result with `metadata.type = "procedural"`.

## Configuration surface

```ts
{
  llm:        { provider: "openai" | "anthropic" | "ollama" | "mock", config: LLMConfig },
  embedder:   { provider: "openai" | "ollama" | "mock",                config: EmbedderConfig },
  vectorStore:{ provider: "sqlite" | "memory" | "qdrant",              config: VectorStoreConfig },
  historyStore?: { provider: "sqlite" | "memory", config: HistoryStoreConfig },
  graphStore?:   { provider: "sqlite" | "memory" | "none", config: GraphStoreConfig },

  enableGraph?:        boolean,
  disableHistory?:     boolean,
  customInstructions?: string,
  scoringWeights?:     { semantic, bm25, entity, graph },
  retry?:              { attempts, baseDelayMs, factor, maxDelayMs },
  entityExtractor?:    { mode: "local" | "llm" | "hybrid",
                          llmFallbackThreshold?, llmForNonLatin? },
}
```

### Storage providers — pick one

| Provider | Persistence | Notes |
|---|---|---|
| `sqlite` (default) | Disk via `bun:sqlite`, single file per store | FTS5 keyword search built in. Set `path` for persistence; omit for `:memory:`. |
| `memory` | None | In-process `Map`-backed; tests / ephemeral. |
| `qdrant` | Remote service | REST-API client; useful when scaling beyond a single process. |

The `VectorStore` / `GraphStore` / `Embedder` / `LLM` / `HistoryManager` interfaces stay open so you can plug in Neo4j, Kuzu, pgvector, etc. without touching the `Memory` class.

### NER strategy

| Mode | When to use |
|---|---|
| `local` (default) | English text, fast, free. Uses compromise + regex augmentations (CJK chunks, emails, URLs, phone numbers, ALL-CAPS acronyms, code identifiers). |
| `llm` | Multi-language inputs where you want maximum recall. Each query/memory pays one LLM call (results are cached per-text). |
| `hybrid` (recommended for production) | Local first; LLM fallback only when local returns nothing OR the text contains non-Latin script (Chinese, Japanese, Korean, Arabic, Cyrillic, Hebrew, Devanagari, Thai). |

## Testing

```bash
bun test                        # all tests
bun test --coverage             # 100% line + function coverage
```

Every external dependency is mocked:
- LLM/Embedder providers → `tests/_helpers/fetch_mock.ts` stubs global `fetch`.
- The `Memory` class is exercised with `MockLLM` (scripted JSON responses) and `MockEmbedder` (deterministic per-character hashing).
- SQLite uses `:memory:`. Tests are hermetic and don't touch the filesystem or network.

## Local-first defaults

- All three persistence layers default to `bun:sqlite` — no native compilation, no external services, single-file databases.
- Pass a `path` config to persist to disk; omit it for `:memory:` mode (useful for tests).
- LLM/Embedder providers default to `localhost:11434` for Ollama, so the entire stack runs offline.
- Pluggable abstractions (`VectorStore`, `GraphStore`, `Embedder`, `LLM`, `HistoryManager`, `EntityExtractor`) are preserved so you can swap in Qdrant / Neo4j / Kuzu / pgvector / Cohere / etc. without touching `Memory`.

## License

Apache-2.0 — same as upstream Mem0.
