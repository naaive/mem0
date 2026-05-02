import { LLM } from "../llms/base";
import { OpenAILLM } from "../llms/openai";
import { AnthropicLLM } from "../llms/anthropic";
import { OllamaLLM } from "../llms/ollama";
import { MockLLM } from "../llms/mock";

import { Embedder } from "../embeddings/base";
import { OpenAIEmbedder } from "../embeddings/openai";
import { OllamaEmbedder } from "../embeddings/ollama";
import { MockEmbedder } from "../embeddings/mock";

import { VectorStore } from "../vector_stores/base";
import { InMemoryVectorStore } from "../vector_stores/memory";
import { QdrantVectorStore } from "../vector_stores/qdrant";

import { HistoryManager } from "../storage/base";
import { SqliteHistoryManager } from "../storage/sqlite";
import { InMemoryHistoryManager } from "../storage/in_memory";

import { GraphStore } from "../graphs/base";
import { InMemoryGraphStore } from "../graphs/in_memory";

import type {
  EmbedderConfig,
  GraphStoreConfig,
  HistoryStoreConfig,
  LLMConfig,
  MemoryConfig,
  VectorStoreConfig,
} from "../types";
import { MemoryConfigSchema } from "../types";

export function createLLM(provider: string, config: LLMConfig): LLM {
  switch (provider) {
    case "openai":
      return new OpenAILLM(config);
    case "anthropic":
      return new AnthropicLLM(config);
    case "ollama":
      return new OllamaLLM(config);
    case "mock":
      return new MockLLM(config);
    default:
      throw new Error(`Unsupported LLM provider: ${provider}`);
  }
}

export function createEmbedder(
  provider: string,
  config: EmbedderConfig,
): Embedder {
  switch (provider) {
    case "openai":
      return new OpenAIEmbedder(config);
    case "ollama":
      return new OllamaEmbedder(config);
    case "mock":
      return new MockEmbedder(config);
    default:
      throw new Error(`Unsupported embedder provider: ${provider}`);
  }
}

export function createVectorStore(
  provider: string,
  config: VectorStoreConfig,
): VectorStore {
  switch (provider) {
    case "memory":
      return new InMemoryVectorStore(config);
    case "qdrant":
      return new QdrantVectorStore(config);
    default:
      throw new Error(`Unsupported vector store provider: ${provider}`);
  }
}

export function createHistoryManager(
  provider: string,
  config: HistoryStoreConfig,
): HistoryManager {
  switch (provider) {
    case "sqlite":
      return new SqliteHistoryManager(config);
    case "memory":
      return new InMemoryHistoryManager();
    default:
      throw new Error(`Unsupported history store provider: ${provider}`);
  }
}

export function createGraphStore(
  provider: string,
  _config: GraphStoreConfig,
): GraphStore | null {
  switch (provider) {
    case "memory":
      return new InMemoryGraphStore();
    case "none":
      return null;
    default:
      throw new Error(`Unsupported graph store provider: ${provider}`);
  }
}

export const DEFAULT_CONFIG: MemoryConfig = {
  llm: { provider: "openai", config: {} },
  embedder: { provider: "openai", config: {} },
  vectorStore: { provider: "memory", config: { collectionName: "mem0" } },
  historyStore: { provider: "sqlite", config: {} },
  graphStore: { provider: "none", config: {} },
};

export function resolveConfig(input: Partial<MemoryConfig>): MemoryConfig {
  const merged: MemoryConfig = {
    ...DEFAULT_CONFIG,
    ...input,
    llm: { ...DEFAULT_CONFIG.llm, ...(input.llm ?? {}) },
    embedder: { ...DEFAULT_CONFIG.embedder, ...(input.embedder ?? {}) },
    vectorStore: {
      ...DEFAULT_CONFIG.vectorStore,
      ...(input.vectorStore ?? {}),
      config: {
        ...DEFAULT_CONFIG.vectorStore.config,
        ...(input.vectorStore?.config ?? {}),
      },
    },
    historyStore: input.historyStore ?? DEFAULT_CONFIG.historyStore,
    graphStore: input.graphStore ?? DEFAULT_CONFIG.graphStore,
  } as MemoryConfig;
  return MemoryConfigSchema.parse(merged);
}
