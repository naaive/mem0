import { describe, expect, test } from "bun:test";
import * as api from "../src/index";

describe("public API surface", () => {
  test("exports Memory, providers, factories, helpers", () => {
    expect(api.Memory).toBeDefined();

    expect(api.LLM).toBeDefined();
    expect(api.OpenAILLM).toBeDefined();
    expect(api.AnthropicLLM).toBeDefined();
    expect(api.OllamaLLM).toBeDefined();
    expect(api.MockLLM).toBeDefined();

    expect(api.Embedder).toBeDefined();
    expect(api.OpenAIEmbedder).toBeDefined();
    expect(api.OllamaEmbedder).toBeDefined();
    expect(api.MockEmbedder).toBeDefined();

    expect(api.VectorStore).toBeDefined();
    expect(api.InMemoryVectorStore).toBeDefined();
    expect(api.QdrantVectorStore).toBeDefined();

    expect(api.HistoryManager).toBeDefined();
    expect(api.SqliteHistoryManager).toBeDefined();
    expect(api.InMemoryHistoryManager).toBeDefined();

    expect(api.createLLM).toBeInstanceOf(Function);
    expect(api.createEmbedder).toBeInstanceOf(Function);
    expect(api.createVectorStore).toBeInstanceOf(Function);
    expect(api.createHistoryManager).toBeInstanceOf(Function);
    expect(api.resolveConfig).toBeInstanceOf(Function);
    expect(api.DEFAULT_CONFIG).toBeDefined();

    expect(api.factRetrievalPrompt).toBeInstanceOf(Function);
    expect(typeof api.UPDATE_MEMORY_PROMPT).toBe("string");
    expect(api.buildUpdateMemoryUserPrompt).toBeInstanceOf(Function);

    expect(api.extractJson).toBeInstanceOf(Function);
    expect(api.safeJsonParse).toBeInstanceOf(Function);
    expect(api.md5).toBeInstanceOf(Function);
    expect(api.uuid).toBeInstanceOf(Function);
    expect(api.nowIso).toBeInstanceOf(Function);
    expect(api.cosineSimilarity).toBeInstanceOf(Function);

    expect(api.normalizeMessages).toBeInstanceOf(Function);
    expect(api.renderTranscript).toBeInstanceOf(Function);
    expect(api.rejectTopLevelEntityParams).toBeInstanceOf(Function);
    expect(api.validateEntityId).toBeInstanceOf(Function);
    expect(api.validateSearchParams).toBeInstanceOf(Function);

    expect(api.MemoryConfigSchema).toBeDefined();
  });
});
