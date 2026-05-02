import { z } from "zod";

export interface Message {
  role: "system" | "user" | "assistant" | string;
  content: string;
}

export interface MemoryItem {
  id: string;
  memory: string;
  hash?: string;
  createdAt?: string;
  updatedAt?: string;
  score?: number;
  metadata?: Record<string, unknown>;
  user_id?: string;
  agent_id?: string;
  run_id?: string;
}

export interface SearchResult {
  results: MemoryItem[];
}

export interface VectorRecord {
  id: string;
  payload: Record<string, unknown>;
  score?: number;
}

export interface SearchFilters {
  user_id?: string;
  agent_id?: string;
  run_id?: string;
  [key: string]: unknown;
}

export interface AddOptions {
  userId?: string;
  agentId?: string;
  runId?: string;
  metadata?: Record<string, unknown>;
  filters?: SearchFilters;
  infer?: boolean;
}

export interface SearchOptions {
  filters?: SearchFilters;
  topK?: number;
  threshold?: number;
}

export interface GetAllOptions {
  filters?: SearchFilters;
  topK?: number;
}

export interface DeleteAllOptions {
  userId?: string;
  agentId?: string;
  runId?: string;
}

export interface HistoryRecord {
  id: number;
  memoryId: string;
  previousValue: string | null;
  newValue: string | null;
  action: "ADD" | "UPDATE" | "DELETE";
  createdAt: string;
  updatedAt: string | null;
  isDeleted: number;
}

export const LLMConfigSchema = z
  .object({
    apiKey: z.string().optional(),
    model: z.string().optional(),
    baseURL: z.string().optional(),
    temperature: z.number().min(0).max(2).optional(),
    maxTokens: z.number().int().positive().optional(),
    topP: z.number().min(0).max(1).optional(),
  })
  .passthrough();
export type LLMConfig = z.infer<typeof LLMConfigSchema> &
  Record<string, unknown>;

export const EmbedderConfigSchema = z.object({
  apiKey: z.string().optional(),
  model: z.string().optional(),
  baseURL: z.string().optional(),
  embeddingDims: z.number().int().positive().optional(),
});
export type EmbedderConfig = z.infer<typeof EmbedderConfigSchema>;

export const VectorStoreConfigSchema = z
  .object({
    collectionName: z.string().optional(),
    dimension: z.number().int().positive().optional(),
    url: z.string().optional(),
    apiKey: z.string().optional(),
    path: z.string().optional(),
  })
  .passthrough();
export type VectorStoreConfig = z.infer<typeof VectorStoreConfigSchema>;

export const HistoryStoreConfigSchema = z.object({
  path: z.string().optional(),
});
export type HistoryStoreConfig = z.infer<typeof HistoryStoreConfigSchema>;

export const ScoringWeightsSchema = z.object({
  semantic: z.number().nonnegative().optional(),
  bm25: z.number().nonnegative().optional(),
  entity: z.number().nonnegative().optional(),
  graph: z.number().nonnegative().optional(),
});
export type ScoringWeights = z.infer<typeof ScoringWeightsSchema>;

export const RetryConfigSchema = z.object({
  attempts: z.number().int().positive().optional(),
  baseDelayMs: z.number().int().nonnegative().optional(),
  factor: z.number().positive().optional(),
  maxDelayMs: z.number().int().positive().optional(),
});
export type RetryConfig = z.infer<typeof RetryConfigSchema>;

export const GraphStoreConfigSchema = z
  .object({
    enabled: z.boolean().optional(),
  })
  .passthrough();
export type GraphStoreConfig = z.infer<typeof GraphStoreConfigSchema>;

export const EntityExtractorConfigSchema = z.object({
  /** "local" | "llm" | "hybrid". Default "local". */
  mode: z.enum(["local", "llm", "hybrid"]).default("local").optional(),
  /** Hybrid: minimum local hits before invoking LLM fallback. */
  llmFallbackThreshold: z.number().int().nonnegative().optional(),
  /** Hybrid: also invoke LLM when text contains non-Latin script. */
  llmForNonLatin: z.boolean().optional(),
});
export type EntityExtractorConfig = z.infer<typeof EntityExtractorConfigSchema>;

export const MemoryConfigSchema = z.object({
  llm: z.object({
    provider: z.enum(["openai", "anthropic", "ollama", "mock"]),
    config: LLMConfigSchema,
  }),
  embedder: z.object({
    provider: z.enum(["openai", "ollama", "mock"]),
    config: EmbedderConfigSchema,
  }),
  vectorStore: z.object({
    provider: z.enum(["sqlite", "memory", "qdrant"]),
    config: VectorStoreConfigSchema,
  }),
  historyStore: z
    .object({
      provider: z.enum(["sqlite", "memory"]).default("sqlite"),
      config: HistoryStoreConfigSchema,
    })
    .optional(),
  graphStore: z
    .object({
      provider: z.enum(["sqlite", "memory", "none"]).default("sqlite"),
      config: GraphStoreConfigSchema,
    })
    .optional(),
  disableHistory: z.boolean().optional(),
  customInstructions: z.string().optional(),
  scoringWeights: ScoringWeightsSchema.optional(),
  retry: RetryConfigSchema.optional(),
  /** Enable graph memory if a graph store is configured. Default false. */
  enableGraph: z.boolean().optional(),
  /** Entity extraction strategy. */
  entityExtractor: EntityExtractorConfigSchema.optional(),
});
export type MemoryConfig = z.infer<typeof MemoryConfigSchema>;

export type FactExtractionEvent = "ADD" | "UPDATE" | "DELETE" | "NONE";
export interface MemoryUpdateAction {
  id: string;
  text: string;
  event: FactExtractionEvent;
  oldMemory?: string;
  attributedTo?: string;
}

export interface ProceduralMemoryOptions {
  agentId?: string;
  userId?: string;
  runId?: string;
  metadata?: Record<string, unknown>;
}
