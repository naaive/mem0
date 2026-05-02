# @mem0/bun

A local-first TypeScript rewrite of Mem0's open-source SDK, built for [Bun](https://bun.com).

This package implements Mem0's core memory engine—LLM-driven fact extraction, smart ADD/UPDATE/DELETE/NONE memory updates, vector retrieval, and full history tracking—in idiomatic TypeScript with no Node dependencies. Storage defaults to in-process: `bun:sqlite` for history and an in-memory cosine-similarity index for vectors. Provider integrations (OpenAI, Anthropic, Ollama, Qdrant) are implemented over `fetch`, with no SDK dependencies.

The package ships with **100% line and function coverage** under `bun test`.

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
  vectorStore: { provider: "memory", config: { dimension: 1536 } },
});

await memory.add(
  [
    { role: "user", content: "Hi, my name is Alice and I love hiking." },
  ],
  { userId: "alice" },
);

const search = await memory.search("What does the user like?", {
  filters: { user_id: "alice" },
});
console.log(search.results);
```

## Public API

```ts
import {
  Memory,
  // LLMs
  LLM, OpenAILLM, AnthropicLLM, OllamaLLM, MockLLM,
  // Embeddings
  Embedder, OpenAIEmbedder, OllamaEmbedder, MockEmbedder,
  // Vector stores
  VectorStore, InMemoryVectorStore, QdrantVectorStore,
  // History
  HistoryManager, SqliteHistoryManager, InMemoryHistoryManager,
  // Factories
  createLLM, createEmbedder, createVectorStore, createHistoryManager,
  resolveConfig, DEFAULT_CONFIG,
  // Prompts (for advanced custom flows)
  factRetrievalPrompt, UPDATE_MEMORY_PROMPT, buildUpdateMemoryUserPrompt,
} from "@mem0/bun";
```

### `Memory`

| Method | Purpose |
|---|---|
| `add(messages, opts)` | Extract facts via LLM and apply ADD/UPDATE/DELETE/NONE. Pass `{ infer: false }` to store messages verbatim. |
| `search(query, opts)` | Embed the query, run cosine similarity over scoped memories, optionally apply `threshold` and `topK`. |
| `get(id)` | Fetch a single memory by id. |
| `getAll(opts)` | List memories scoped by `filters: { user_id, agent_id, run_id }`. |
| `update(id, text)` | Rewrite a memory. Re-embeds and writes a history row. |
| `delete(id)` | Remove a memory and write a history row. |
| `deleteAll(opts)` | Remove all memories scoped by entity ids. |
| `getHistory(id)` | Read the history log for a memory. |
| `reset()` | Drop the vector collection and history. |
| `close()` | Release the history backend (e.g. SQLite). |

`opts` always accepts at least one of `userId`, `agentId`, `runId` (camelCase) or the same in `filters` (snake_case). Top-level entity ids inside `search`/`getAll` `config` are rejected to avoid filter typos.

## Architecture

```
src/
├── memory/            Memory class — fact extraction + memory update orchestration
├── llms/              base + openai + anthropic + ollama + mock
├── embeddings/        base + openai + ollama + mock
├── vector_stores/     base + in-memory + qdrant
├── storage/           base + sqlite (bun:sqlite) + in-memory history
├── prompts/           FACT_RETRIEVAL + UPDATE_MEMORY prompts
├── config/            factory.ts (resolveConfig + provider factories)
├── utils/             json, hash, message normalization
└── types/             zod schemas + TypeScript types
```

The default flow on `add()`:

1. **Normalize** input messages, validate entity ids, build a transcript.
2. **Extract facts** with the LLM (FACT_RETRIEVAL prompt, JSON output).
3. **Retrieve scoped existing memories** via the vector store.
4. **Diff** existing vs. new with the LLM (UPDATE_MEMORY prompt) → action list.
5. **Apply** ADD/UPDATE/DELETE actions, re-embed where needed, write history.

`infer: false` short-circuits the LLM and stores each non-system message verbatim.

## Testing

```bash
bun test                   # run all tests
bun test --coverage        # 100% line + function coverage
```

The test suite mocks every external dependency:

- LLM/Embedder providers → global `fetch` is stubbed by `tests/_helpers/fetch_mock.ts`.
- The `Memory` class is exercised end-to-end via the `MockLLM` and `MockEmbedder`, which produce deterministic responses and embeddings.
- SQLite uses `:memory:`, so tests are hermetic and don't touch the filesystem.

## Local-first design

- `historyStore.provider: "sqlite"` (the default) uses `bun:sqlite`, which ships with Bun—no native module compilation, no external service.
- `vectorStore.provider: "memory"` is purely in-process and supports the full search/list/delete API used by `Memory`.
- LLM/Embedder providers default to `localhost:11434` for Ollama, so you can run the entire stack locally with `ollama serve` + a downloaded model. Cloud providers are opt-in.

## License

Apache-2.0 — same as upstream Mem0.
