import {
  type AddOptions,
  type DeleteAllOptions,
  type GetAllOptions,
  type HistoryRecord,
  type MemoryConfig,
  type MemoryItem,
  type Message,
  type SearchFilters,
  type SearchOptions,
  type SearchResult,
} from "../types";
import { LLM } from "../llms/base";
import { Embedder } from "../embeddings/base";
import { VectorStore } from "../vector_stores/base";
import { HistoryManager } from "../storage/base";
import {
  createEmbedder,
  createHistoryManager,
  createLLM,
  createVectorStore,
  resolveConfig,
} from "../config/factory";
import {
  buildUpdateMemoryUserPrompt,
  factRetrievalPrompt,
  UPDATE_MEMORY_PROMPT,
} from "../prompts";
import { extractJson } from "../utils/json";
import { md5, nowIso, uuid } from "../utils/hash";
import {
  normalizeMessages,
  rejectTopLevelEntityParams,
  renderTranscript,
  validateEntityId,
  validateSearchParams,
} from "../utils/messages";

const RESERVED_PAYLOAD_KEYS = new Set([
  "data",
  "hash",
  "createdAt",
  "updatedAt",
  "user_id",
  "agent_id",
  "run_id",
]);

export class Memory {
  private readonly config: MemoryConfig;
  private readonly llm: LLM;
  private readonly embedder: Embedder;
  private readonly vectorStore: VectorStore;
  private readonly history: HistoryManager;
  private readonly historyEnabled: boolean;
  private initialized = false;

  constructor(config: Partial<MemoryConfig> = {}) {
    this.config = resolveConfig(config);
    this.llm = createLLM(this.config.llm.provider, this.config.llm.config);
    this.embedder = createEmbedder(
      this.config.embedder.provider,
      this.config.embedder.config,
    );
    this.vectorStore = createVectorStore(
      this.config.vectorStore.provider,
      this.config.vectorStore.config,
    );
    this.historyEnabled = !this.config.disableHistory;
    this.history = this.historyEnabled
      ? createHistoryManager(
          this.config.historyStore!.provider,
          this.config.historyStore!.config,
        )
      : createHistoryManager("memory", {});
  }

  static async create(config: Partial<MemoryConfig> = {}): Promise<Memory> {
    const m = new Memory(config);
    await m.initialize();
    return m;
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;
    await this.vectorStore.initialize();
    this.initialized = true;
  }

  /** Build a session-scope filter set with only defined entity ids. */
  private buildFilters(opts: {
    userId?: string;
    agentId?: string;
    runId?: string;
    extra?: SearchFilters;
  }): SearchFilters {
    const out: SearchFilters = { ...(opts.extra ?? {}) };
    const u = validateEntityId(opts.userId, "userId");
    const a = validateEntityId(opts.agentId, "agentId");
    const r = validateEntityId(opts.runId, "runId");
    if (u) out.user_id = u;
    if (a) out.agent_id = a;
    if (r) out.run_id = r;
    return out;
  }

  private async extractFacts(
    transcript: string,
    customInstructions?: string,
  ): Promise<string[]> {
    const system = factRetrievalPrompt();
    const userBody = customInstructions
      ? `${customInstructions}\n\n${transcript}`
      : transcript;
    const raw = await this.llm.generateResponse(
      [
        { role: "system", content: system },
        { role: "user", content: userBody },
      ],
      { responseFormat: "json_object" },
    );
    let parsed: { facts?: unknown };
    try {
      parsed = JSON.parse(extractJson(raw));
    } catch {
      return [];
    }
    if (!parsed || !Array.isArray(parsed.facts)) return [];
    return parsed.facts.filter(
      (f): f is string => typeof f === "string" && f.trim().length > 0,
    );
  }

  private async decideUpdates(
    existing: Array<{ id: string; text: string }>,
    facts: string[],
  ): Promise<
    Array<{ id: string; text: string; event: string; oldMemory?: string }>
  > {
    if (facts.length === 0) return [];
    const userPrompt = buildUpdateMemoryUserPrompt(
      existing,
      facts,
      this.config.customInstructions,
    );
    const raw = await this.llm.generateResponse(
      [
        { role: "system", content: UPDATE_MEMORY_PROMPT },
        { role: "user", content: userPrompt },
      ],
      { responseFormat: "json_object" },
    );
    let parsed: { memory?: unknown };
    try {
      parsed = JSON.parse(extractJson(raw));
    } catch {
      return [];
    }
    if (!parsed || !Array.isArray(parsed.memory)) return [];
    const out: Array<{
      id: string;
      text: string;
      event: string;
      oldMemory?: string;
    }> = [];
    for (const item of parsed.memory as unknown[]) {
      if (typeof item !== "object" || item === null) continue;
      const rec = item as Record<string, unknown>;
      const id = typeof rec.id === "string" ? rec.id : String(rec.id ?? "");
      const text = typeof rec.text === "string" ? rec.text : "";
      const event =
        typeof rec.event === "string" ? rec.event.toUpperCase() : "NONE";
      const oldMemory =
        typeof rec.old_memory === "string" ? rec.old_memory : undefined;
      if (!text || !["ADD", "UPDATE", "DELETE", "NONE"].includes(event)) {
        continue;
      }
      out.push({ id, text, event, oldMemory });
    }
    return out;
  }

  async add(
    messages: string | Message[],
    config: AddOptions = {},
  ): Promise<SearchResult> {
    await this.initialize();
    const parsed = normalizeMessages(messages);
    const metadata = { ...(config.metadata ?? {}) };
    const filters = this.buildFilters({
      userId: config.userId,
      agentId: config.agentId,
      runId: config.runId,
      extra: config.filters,
    });

    if (!filters.user_id && !filters.agent_id && !filters.run_id) {
      throw new Error(
        "One of userId, agentId, or runId is required to add memories",
      );
    }
    if (filters.user_id) metadata.user_id = filters.user_id;
    if (filters.agent_id) metadata.agent_id = filters.agent_id;
    if (filters.run_id) metadata.run_id = filters.run_id;

    const transcript = renderTranscript(parsed);
    const infer = config.infer !== false;

    if (!infer) {
      const out: MemoryItem[] = [];
      for (const m of parsed) {
        if (m.role === "system") continue;
        const id = await this.persistMemory(m.content, metadata, filters);
        out.push({ id, memory: m.content, metadata: { event: "ADD" } });
      }
      return { results: out };
    }

    const facts = await this.extractFacts(
      transcript,
      this.config.customInstructions,
    );
    if (facts.length === 0) return { results: [] };

    // Retrieve existing memories scoped to filters for the LLM to decide diffs.
    const queryEmbedding = await this.embedder.embed(transcript);
    const existingResults = await this.vectorStore.search(
      queryEmbedding,
      10,
      filters,
    );
    const idMap = new Map<string, string>();
    const existing = existingResults.map((r, idx) => {
      const tempId = String(idx);
      idMap.set(tempId, r.id);
      return { id: tempId, text: String(r.payload.data ?? "") };
    });

    const decisions = await this.decideUpdates(existing, facts);

    const results: MemoryItem[] = [];
    for (const d of decisions) {
      const realId = idMap.get(d.id);
      try {
        if (d.event === "ADD") {
          const id = await this.persistMemory(d.text, metadata, filters);
          results.push({ id, memory: d.text, metadata: { event: "ADD" } });
        } else if (d.event === "UPDATE" && realId) {
          await this.applyUpdate(realId, d.text);
          results.push({
            id: realId,
            memory: d.text,
            metadata: { event: "UPDATE", previousMemory: d.oldMemory },
          });
        } else if (d.event === "DELETE" && realId) {
          await this.applyDelete(realId);
          results.push({
            id: realId,
            memory: d.text,
            metadata: { event: "DELETE" },
          });
        }
        // NONE → skip
      } catch (err) {
        // Surface persistence errors with context but do not abort the batch.
        results.push({
          id: realId ?? "",
          memory: d.text,
          metadata: {
            event: "ERROR",
            error: err instanceof Error ? err.message : String(err),
          },
        });
      }
    }

    return { results };
  }

  private async persistMemory(
    text: string,
    metadata: Record<string, unknown>,
    filters: SearchFilters,
  ): Promise<string> {
    const id = uuid();
    const embedding = await this.embedder.embed(text);
    const created = nowIso();
    const payload: Record<string, unknown> = {
      ...metadata,
      data: text,
      hash: md5(text),
      createdAt: created,
    };
    if (filters.user_id) payload.user_id = filters.user_id;
    if (filters.agent_id) payload.agent_id = filters.agent_id;
    if (filters.run_id) payload.run_id = filters.run_id;
    await this.vectorStore.insert([embedding], [id], [payload]);
    if (this.historyEnabled) {
      await this.history.addHistory({
        memoryId: id,
        previousValue: null,
        newValue: text,
        action: "ADD",
        createdAt: created,
      });
    }
    return id;
  }

  private async applyUpdate(memoryId: string, text: string): Promise<void> {
    const existing = await this.vectorStore.get(memoryId);
    if (!existing) {
      throw new Error(`Memory with id ${memoryId} not found`);
    }
    const previous = String(existing.payload.data ?? "");
    const embedding = await this.embedder.embed(text);
    const updated = nowIso();
    const newPayload: Record<string, unknown> = {
      ...existing.payload,
      data: text,
      hash: md5(text),
      updatedAt: updated,
    };
    await this.vectorStore.update(memoryId, embedding, newPayload);
    if (this.historyEnabled) {
      await this.history.addHistory({
        memoryId,
        previousValue: previous,
        newValue: text,
        action: "UPDATE",
        createdAt:
          (existing.payload.createdAt as string | undefined) ?? updated,
        updatedAt: updated,
      });
    }
  }

  private async applyDelete(memoryId: string): Promise<void> {
    const existing = await this.vectorStore.get(memoryId);
    if (!existing) {
      throw new Error(`Memory with id ${memoryId} not found`);
    }
    const previous = String(existing.payload.data ?? "");
    await this.vectorStore.delete(memoryId);
    if (this.historyEnabled) {
      await this.history.addHistory({
        memoryId,
        previousValue: previous,
        newValue: null,
        action: "DELETE",
        isDeleted: 1,
      });
    }
  }

  private toMemoryItem(rec: {
    id: string;
    payload: Record<string, unknown>;
    score?: number;
  }): MemoryItem {
    const meta: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(rec.payload)) {
      if (!RESERVED_PAYLOAD_KEYS.has(key)) meta[key] = value;
    }
    const item: MemoryItem = {
      id: rec.id,
      memory: String(rec.payload.data ?? ""),
      hash: rec.payload.hash as string | undefined,
      createdAt: rec.payload.createdAt as string | undefined,
      updatedAt: rec.payload.updatedAt as string | undefined,
      metadata: meta,
    };
    if (rec.score !== undefined) item.score = rec.score;
    if (rec.payload.user_id) item.user_id = String(rec.payload.user_id);
    if (rec.payload.agent_id) item.agent_id = String(rec.payload.agent_id);
    if (rec.payload.run_id) item.run_id = String(rec.payload.run_id);
    return item;
  }

  async get(memoryId: string): Promise<MemoryItem | null> {
    await this.initialize();
    const rec = await this.vectorStore.get(memoryId);
    if (!rec) return null;
    return this.toMemoryItem(rec);
  }

  async search(
    query: string,
    config: SearchOptions = {},
  ): Promise<SearchResult> {
    await this.initialize();
    rejectTopLevelEntityParams(
      config as unknown as Record<string, unknown>,
      "search",
    );
    validateSearchParams(config.threshold, config.topK);
    const topK = config.topK ?? 20;
    const threshold = config.threshold ?? 0;

    const filters: SearchFilters = {};
    const f = config.filters ?? {};
    const u = validateEntityId(f.user_id, "user_id");
    const a = validateEntityId(f.agent_id, "agent_id");
    const r = validateEntityId(f.run_id, "run_id");
    if (u) filters.user_id = u;
    if (a) filters.agent_id = a;
    if (r) filters.run_id = r;
    for (const [k, v] of Object.entries(f)) {
      if (
        k !== "user_id" &&
        k !== "agent_id" &&
        k !== "run_id" &&
        v !== undefined
      ) {
        filters[k] = v;
      }
    }
    if (!filters.user_id && !filters.agent_id && !filters.run_id) {
      throw new Error(
        "filters must contain at least one of user_id, agent_id, run_id",
      );
    }

    const embedding = await this.embedder.embed(query);
    const records = await this.vectorStore.search(
      embedding,
      Math.max(topK, 1),
      filters,
    );
    const results = records
      .filter((r) => (r.score ?? 0) >= threshold && r.payload.data)
      .slice(0, topK)
      .map((r) => this.toMemoryItem(r));
    return { results };
  }

  async getAll(config: GetAllOptions = {}): Promise<SearchResult> {
    await this.initialize();
    rejectTopLevelEntityParams(
      config as unknown as Record<string, unknown>,
      "getAll",
    );
    validateSearchParams(undefined, config.topK);
    const topK = config.topK ?? 100;
    const filters: SearchFilters = {};
    const f = config.filters ?? {};
    const u = validateEntityId(f.user_id, "user_id");
    const a = validateEntityId(f.agent_id, "agent_id");
    const r = validateEntityId(f.run_id, "run_id");
    if (u) filters.user_id = u;
    if (a) filters.agent_id = a;
    if (r) filters.run_id = r;
    if (!filters.user_id && !filters.agent_id && !filters.run_id) {
      throw new Error(
        "filters must contain at least one of user_id, agent_id, run_id",
      );
    }
    const records = await this.vectorStore.list(filters, topK);
    return { results: records.map((r) => this.toMemoryItem(r)) };
  }

  async update(memoryId: string, text: string): Promise<{ message: string }> {
    await this.initialize();
    if (!memoryId || typeof memoryId !== "string") {
      throw new Error("memoryId is required");
    }
    if (!text || typeof text !== "string") {
      throw new Error("text is required");
    }
    await this.applyUpdate(memoryId, text);
    return { message: "Memory updated successfully" };
  }

  async delete(memoryId: string): Promise<{ message: string }> {
    await this.initialize();
    if (!memoryId || typeof memoryId !== "string") {
      throw new Error("memoryId is required");
    }
    await this.applyDelete(memoryId);
    return { message: "Memory deleted successfully" };
  }

  async deleteAll(config: DeleteAllOptions = {}): Promise<{ message: string }> {
    await this.initialize();
    const filters = this.buildFilters(config);
    if (!filters.user_id && !filters.agent_id && !filters.run_id) {
      throw new Error(
        "deleteAll requires at least one of userId, agentId, runId. Use reset() to clear all memories.",
      );
    }
    const records = await this.vectorStore.list(filters, 10000);
    for (const r of records) {
      await this.applyDelete(r.id);
    }
    return { message: "Memories deleted successfully" };
  }

  async getHistory(memoryId: string): Promise<HistoryRecord[]> {
    await this.initialize();
    return this.history.getHistory(memoryId);
  }

  async reset(): Promise<void> {
    await this.initialize();
    await this.vectorStore.deleteCollection();
    await this.history.reset();
    this.initialized = false;
    await this.initialize();
  }

  async close(): Promise<void> {
    await this.history.close();
  }
}
