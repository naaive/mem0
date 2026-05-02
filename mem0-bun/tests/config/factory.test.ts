import { describe, expect, test } from "bun:test";
import {
  createEmbedder,
  createGraphStore,
  createHistoryManager,
  createLLM,
  createVectorStore,
  DEFAULT_CONFIG,
  resolveConfig,
} from "../../src/config/factory";
import { InMemoryGraphStore } from "../../src/graphs/in_memory";
import { MockLLM } from "../../src/llms/mock";
import { OpenAILLM } from "../../src/llms/openai";
import { AnthropicLLM } from "../../src/llms/anthropic";
import { OllamaLLM } from "../../src/llms/ollama";
import { MockEmbedder } from "../../src/embeddings/mock";
import { OpenAIEmbedder } from "../../src/embeddings/openai";
import { OllamaEmbedder } from "../../src/embeddings/ollama";
import { InMemoryVectorStore } from "../../src/vector_stores/memory";
import { QdrantVectorStore } from "../../src/vector_stores/qdrant";
import { SqliteHistoryManager } from "../../src/storage/sqlite";
import { InMemoryHistoryManager } from "../../src/storage/in_memory";

describe("createLLM", () => {
  test("openai", () => {
    expect(createLLM("openai", { apiKey: "k" })).toBeInstanceOf(OpenAILLM);
  });
  test("anthropic", () => {
    expect(createLLM("anthropic", { apiKey: "k" })).toBeInstanceOf(
      AnthropicLLM,
    );
  });
  test("ollama", () => {
    expect(createLLM("ollama", {})).toBeInstanceOf(OllamaLLM);
  });
  test("mock", () => {
    expect(createLLM("mock", {})).toBeInstanceOf(MockLLM);
  });
  test("unknown throws", () => {
    expect(() => createLLM("???", {})).toThrow(/Unsupported LLM/);
  });
});

describe("createEmbedder", () => {
  test("openai", () => {
    expect(createEmbedder("openai", { apiKey: "k" })).toBeInstanceOf(
      OpenAIEmbedder,
    );
  });
  test("ollama", () => {
    expect(createEmbedder("ollama", {})).toBeInstanceOf(OllamaEmbedder);
  });
  test("mock", () => {
    expect(createEmbedder("mock", {})).toBeInstanceOf(MockEmbedder);
  });
  test("unknown throws", () => {
    expect(() => createEmbedder("???", {})).toThrow(/Unsupported embedder/);
  });
});

describe("createVectorStore", () => {
  test("memory", () => {
    expect(createVectorStore("memory", {})).toBeInstanceOf(
      InMemoryVectorStore,
    );
  });
  test("qdrant", () => {
    expect(
      createVectorStore("qdrant", { dimension: 4 }),
    ).toBeInstanceOf(QdrantVectorStore);
  });
  test("unknown throws", () => {
    expect(() => createVectorStore("???", {})).toThrow(
      /Unsupported vector store/,
    );
  });
});

describe("createHistoryManager", () => {
  test("sqlite", () => {
    const h = createHistoryManager("sqlite", {});
    expect(h).toBeInstanceOf(SqliteHistoryManager);
  });
  test("memory", () => {
    expect(createHistoryManager("memory", {})).toBeInstanceOf(
      InMemoryHistoryManager,
    );
  });
  test("unknown throws", () => {
    expect(() => createHistoryManager("???", {})).toThrow(
      /Unsupported history/,
    );
  });
});

describe("createGraphStore", () => {
  test("memory returns InMemoryGraphStore", () => {
    expect(createGraphStore("memory", {})).toBeInstanceOf(InMemoryGraphStore);
  });
  test("none returns null", () => {
    expect(createGraphStore("none", {})).toBeNull();
  });
  test("unknown throws", () => {
    expect(() => createGraphStore("???", {})).toThrow(/Unsupported graph/);
  });
});

describe("resolveConfig", () => {
  test("merges defaults", () => {
    const cfg = resolveConfig({
      llm: { provider: "mock", config: {} },
      embedder: { provider: "mock", config: {} },
      vectorStore: { provider: "memory", config: {} },
    });
    expect(cfg.vectorStore.config.collectionName).toBe("mem0");
    expect(cfg.historyStore?.provider).toBe("sqlite");
  });

  test("preserves user historyStore", () => {
    const cfg = resolveConfig({
      llm: { provider: "mock", config: {} },
      embedder: { provider: "mock", config: {} },
      vectorStore: { provider: "memory", config: {} },
      historyStore: { provider: "memory", config: {} },
    });
    expect(cfg.historyStore?.provider).toBe("memory");
  });

  test("uses pure defaults when input is empty", () => {
    const cfg = resolveConfig({});
    expect(cfg.llm.provider).toBe("openai");
    expect(cfg.embedder.provider).toBe("openai");
    expect(cfg.vectorStore.provider).toBe("memory");
  });

  test("rejects invalid provider", () => {
    expect(() =>
      resolveConfig({
        llm: { provider: "nope" as "mock", config: {} },
      }),
    ).toThrow();
  });

  test("DEFAULT_CONFIG is sane", () => {
    expect(DEFAULT_CONFIG.vectorStore.provider).toBe("memory");
  });
});
