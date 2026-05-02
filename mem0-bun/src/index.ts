export { Memory } from "./memory/memory";

export { LLM } from "./llms/base";
export { OpenAILLM } from "./llms/openai";
export { AnthropicLLM } from "./llms/anthropic";
export { OllamaLLM } from "./llms/ollama";
export { MockLLM, type MockLLMResponder } from "./llms/mock";

export { Embedder } from "./embeddings/base";
export { OpenAIEmbedder } from "./embeddings/openai";
export { OllamaEmbedder } from "./embeddings/ollama";
export { MockEmbedder } from "./embeddings/mock";

export { VectorStore } from "./vector_stores/base";
export { InMemoryVectorStore } from "./vector_stores/memory";
export { QdrantVectorStore } from "./vector_stores/qdrant";

export { HistoryManager } from "./storage/base";
export { SqliteHistoryManager } from "./storage/sqlite";
export { InMemoryHistoryManager } from "./storage/in_memory";

export {
  createEmbedder,
  createHistoryManager,
  createLLM,
  createVectorStore,
  resolveConfig,
  DEFAULT_CONFIG,
} from "./config/factory";

export {
  factRetrievalPrompt,
  UPDATE_MEMORY_PROMPT,
  buildUpdateMemoryUserPrompt,
} from "./prompts";

export { extractJson, safeJsonParse } from "./utils/json";
export { md5, uuid, nowIso, cosineSimilarity } from "./utils/hash";
export {
  normalizeMessages,
  renderTranscript,
  rejectTopLevelEntityParams,
  validateEntityId,
  validateSearchParams,
} from "./utils/messages";

export type {
  AddOptions,
  DeleteAllOptions,
  EmbedderConfig,
  FactExtractionEvent,
  GetAllOptions,
  HistoryRecord,
  HistoryStoreConfig,
  LLMConfig,
  MemoryConfig,
  MemoryItem,
  MemoryUpdateAction,
  Message,
  SearchFilters,
  SearchOptions,
  SearchResult,
  VectorRecord,
  VectorStoreConfig,
} from "./types";

export { MemoryConfigSchema } from "./types";
