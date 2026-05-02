import {
  type AddOptions,
  type DeleteAllOptions,
  type GetAllOptions,
  type HistoryRecord,
  type MemoryConfig,
  type MemoryItem,
  type Message,
  type ProceduralMemoryOptions,
  type SearchFilters,
  type SearchOptions,
  type SearchResult,
} from "../types";
import { LLM } from "../llms/base";
import { Embedder } from "../embeddings/base";
import { VectorStore } from "../vector_stores/base";
import { HistoryManager } from "../storage/base";
import { GraphStore } from "../graphs/base";
import {
  createEmbedder,
  createGraphStore,
  createHistoryManager,
  createLLM,
  createVectorStore,
  resolveConfig,
} from "../config/factory";
import {
  ADDITIVE_EXTRACTION_PROMPT,
  AGENT_CONTEXT_SUFFIX,
  TRIPLE_EXTRACTION_PROMPT,
  PROCEDURAL_MEMORY_PROMPT,
  buildAdditiveExtractionUserPrompt,
  buildTripleExtractionUserPrompt,
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
import { lemmatizeForBm25 } from "../utils/lemmatization";
import {
  extractEntities,
  extractEntitiesBatch,
  type Entity,
} from "../utils/entity_extraction";
import { getBm25Params, normalizeBm25 } from "../utils/bm25";
import {
  entityBoostFor,
  scoreAndRank,
  type Candidate,
} from "../utils/scoring";
import { withRetry } from "../utils/retry";

const RESERVED_PAYLOAD_KEYS = new Set([
  "data",
  "hash",
  "createdAt",
  "updatedAt",
  "user_id",
  "agent_id",
  "run_id",
  "textLemmatized",
  "attributedTo",
]);

const ENTITY_BOOST_WEIGHT = 0.3;

interface ExtractedMemory {
  id: string;
  text: string;
  event: "ADD" | "UPDATE" | "DELETE" | "NONE";
  oldMemory?: string;
  attributedTo?: string;
}

export class Memory {
  private readonly config: MemoryConfig;
  private readonly llm: LLM;
  private readonly embedder: Embedder;
  private readonly vectorStore: VectorStore;
  private readonly history: HistoryManager;
  private readonly historyEnabled: boolean;
  private readonly entityStore: VectorStore | null;
  private readonly graphStore: GraphStore | null;
  private readonly graphEnabled: boolean;
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

    // Entity store reuses the same vector backend with a sibling collection.
    const baseColl =
      this.config.vectorStore.config.collectionName ?? "mem0_default";
    this.entityStore = createVectorStore(this.config.vectorStore.provider, {
      ...this.config.vectorStore.config,
      collectionName: `${baseColl}_entities`,
    });

    this.historyEnabled = !this.config.disableHistory;
    this.history = this.historyEnabled
      ? createHistoryManager(
          this.config.historyStore!.provider,
          this.config.historyStore!.config,
        )
      : createHistoryManager("memory", {});

    this.graphStore = createGraphStore(
      this.config.graphStore?.provider ?? "none",
      this.config.graphStore?.config ?? {},
    );
    this.graphEnabled = !!this.config.enableGraph && this.graphStore !== null;
  }

  static async create(config: Partial<MemoryConfig> = {}): Promise<Memory> {
    const m = new Memory(config);
    await m.initialize();
    return m;
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;
    await this.vectorStore.initialize();
    if (this.entityStore) await this.entityStore.initialize();
    if (this.graphStore) await this.graphStore.initialize();
    this.initialized = true;
  }

  // ---------------- helpers ----------------

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

  private sessionScopeFilters(payload: Record<string, unknown>): SearchFilters {
    const f: SearchFilters = {};
    if (payload.user_id) f.user_id = String(payload.user_id);
    if (payload.agent_id) f.agent_id = String(payload.agent_id);
    if (payload.run_id) f.run_id = String(payload.run_id);
    return f;
  }

  private async embedRetried(text: string): Promise<number[]> {
    return withRetry(() => this.embedder.embed(text), this.config.retry);
  }

  private async embedBatchRetried(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    return withRetry(
      () => this.embedder.embedBatch(texts),
      this.config.retry,
    );
  }

  private async llmRetried(
    messages: Message[],
    options: { responseFormat?: "json_object" | "text" } = {},
  ): Promise<string> {
    return withRetry(
      () => this.llm.generateResponse(messages, options),
      this.config.retry,
    );
  }

  // ---------------- add ----------------

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

    return this.addWithInference(transcript, metadata, filters, parsed);
  }

  /**
   * V3 phased pipeline:
   *  1. Pull last K messages for context.
   *  2. Embed transcript + retrieve scoped existing memories.
   *  3. Single LLM call: ADDITIVE_EXTRACTION_PROMPT.
   *  4. Hash dedup + batch embed memory texts.
   *  5. Apply ADD/UPDATE/DELETE actions, batch where possible.
   *  6. Entity store linking.
   *  7. Graph triple extraction + storage (when graph enabled).
   */
  private async addWithInference(
    transcript: string,
    metadata: Record<string, unknown>,
    filters: SearchFilters,
    rawMessages: Message[],
  ): Promise<SearchResult> {
    const isAgentScoped = !!filters.agent_id && !filters.user_id;

    // Phase 1: Retrieve existing memories.
    const queryEmbedding = await this.embedRetried(transcript);
    const existingResults = await this.vectorStore.search(
      queryEmbedding,
      10,
      filters,
    );
    const idMap = new Map<string, string>();
    const existing: Array<{ id: string; text: string }> = existingResults.map(
      (r, idx) => {
        const tempId = String(idx);
        idMap.set(tempId, r.id);
        return { id: tempId, text: String(r.payload.data ?? "") };
      },
    );

    // Phase 2: Single-call additive extraction.
    const systemPrompt = isAgentScoped
      ? ADDITIVE_EXTRACTION_PROMPT + AGENT_CONTEXT_SUFFIX
      : ADDITIVE_EXTRACTION_PROMPT;
    const userPrompt = buildAdditiveExtractionUserPrompt({
      existingMemories: existing,
      newMessages: transcript,
      customInstructions: this.config.customInstructions,
    });
    let raw: string;
    try {
      raw = await this.llmRetried(
        [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
        { responseFormat: "json_object" },
      );
    } catch {
      return { results: [] };
    }

    const decisions = this.parseExtractionResponse(raw);
    if (decisions.length === 0) return { results: [] };

    // Phase 3: Batch embed ADD/UPDATE texts upfront.
    const textsToEmbed = decisions
      .filter((d) => d.event === "ADD" || d.event === "UPDATE")
      .map((d) => d.text);
    let embedMap = new Map<string, number[]>();
    if (textsToEmbed.length > 0) {
      try {
        const vectors = await this.embedBatchRetried(textsToEmbed);
        for (let i = 0; i < textsToEmbed.length; i++) {
          embedMap.set(textsToEmbed[i]!, vectors[i]!);
        }
      } catch {
        // Embedding failed entirely — fall back to per-text retries
        // inside persistMemory/applyUpdate.
        embedMap = new Map();
      }
    }

    // Phase 4: Apply actions.
    const results: MemoryItem[] = [];
    const newMemoryRecords: Array<{ id: string; text: string }> = [];
    for (const d of decisions) {
      const realId = idMap.get(d.id);
      try {
        if (d.event === "ADD") {
          const id = await this.persistMemory(
            d.text,
            { ...metadata, ...(d.attributedTo && { attributedTo: d.attributedTo }) },
            filters,
            embedMap.get(d.text),
          );
          results.push({
            id,
            memory: d.text,
            metadata: {
              event: "ADD",
              ...(d.attributedTo && { attributedTo: d.attributedTo }),
            },
          });
          newMemoryRecords.push({ id, text: d.text });
        } else if (d.event === "UPDATE" && realId) {
          await this.applyUpdate(realId, d.text, embedMap.get(d.text));
          results.push({
            id: realId,
            memory: d.text,
            metadata: { event: "UPDATE", previousMemory: d.oldMemory },
          });
          newMemoryRecords.push({ id: realId, text: d.text });
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

    // Phase 5: Entity-store linking (best effort).
    if (newMemoryRecords.length > 0 && this.entityStore) {
      try {
        await this.linkEntities(newMemoryRecords, filters);
      } catch {
        /* non-fatal */
      }
    }

    // Phase 6: Graph triples (best effort, only when enabled).
    if (this.graphEnabled && newMemoryRecords.length > 0) {
      try {
        await this.linkGraphTriples(newMemoryRecords, filters);
      } catch {
        /* non-fatal */
      }
    }

    // Phase 7: Save raw messages to history for context in future calls.
    if (this.historyEnabled) {
      for (const m of rawMessages) {
        if (m.role === "system") continue;
        try {
          await this.history.addHistory({
            memoryId: `__msg_${uuid()}`,
            previousValue: null,
            newValue: `${m.role}: ${m.content}`,
            action: "ADD",
          });
        } catch {
          /* non-fatal */
        }
      }
    }

    return { results };
  }

  private parseExtractionResponse(raw: string): ExtractedMemory[] {
    let parsed: { memory?: unknown };
    try {
      parsed = JSON.parse(extractJson(raw));
    } catch {
      return [];
    }
    if (!parsed || !Array.isArray(parsed.memory)) return [];
    const out: ExtractedMemory[] = [];
    const seenHashes = new Set<string>();
    for (const item of parsed.memory as unknown[]) {
      if (typeof item !== "object" || item === null) continue;
      const rec = item as Record<string, unknown>;
      const id = typeof rec.id === "string" ? rec.id : String(rec.id ?? "");
      const text = typeof rec.text === "string" ? rec.text.trim() : "";
      const event =
        typeof rec.event === "string" ? rec.event.toUpperCase() : "NONE";
      const oldMemory =
        typeof rec.old_memory === "string" ? rec.old_memory : undefined;
      const attributedTo =
        typeof rec.attributed_to === "string" ? rec.attributed_to : undefined;
      if (!text) continue;
      if (!["ADD", "UPDATE", "DELETE", "NONE"].includes(event)) continue;
      // dedup by content hash for ADD/UPDATE so the same fact isn't applied twice
      if (event === "ADD" || event === "UPDATE") {
        const h = md5(text);
        if (seenHashes.has(h)) continue;
        seenHashes.add(h);
      }
      out.push({
        id,
        text,
        event: event as ExtractedMemory["event"],
        oldMemory,
        attributedTo,
      });
    }
    return out;
  }

  private async persistMemory(
    text: string,
    metadata: Record<string, unknown>,
    filters: SearchFilters,
    cachedEmbedding?: number[],
  ): Promise<string> {
    const id = uuid();
    const embedding = cachedEmbedding ?? (await this.embedRetried(text));
    const created = nowIso();
    const payload: Record<string, unknown> = {
      ...metadata,
      data: text,
      hash: md5(text),
      textLemmatized: lemmatizeForBm25(text),
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

  private async applyUpdate(
    memoryId: string,
    text: string,
    cachedEmbedding?: number[],
  ): Promise<void> {
    const existing = await this.vectorStore.get(memoryId);
    if (!existing) {
      throw new Error(`Memory with id ${memoryId} not found`);
    }
    const previous = String(existing.payload.data ?? "");
    const embedding = cachedEmbedding ?? (await this.embedRetried(text));
    const updated = nowIso();
    const newPayload: Record<string, unknown> = {
      ...existing.payload,
      data: text,
      hash: md5(text),
      textLemmatized: lemmatizeForBm25(text),
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
    // Re-link entities & graph triples.
    if (this.entityStore) {
      const sessionFilters = this.sessionScopeFilters(newPayload);
      try {
        await this.removeMemoryFromEntityStore(memoryId, sessionFilters);
        await this.linkEntities([{ id: memoryId, text }], sessionFilters);
      } catch {
        /* non-fatal */
      }
    }
    if (this.graphEnabled && this.graphStore) {
      const sessionFilters = this.sessionScopeFilters(newPayload);
      try {
        await this.graphStore.deleteByMemoryId(memoryId, sessionFilters);
        await this.linkGraphTriples([{ id: memoryId, text }], sessionFilters);
      } catch {
        /* non-fatal */
      }
    }
  }

  private async applyDelete(memoryId: string): Promise<void> {
    const existing = await this.vectorStore.get(memoryId);
    if (!existing) {
      throw new Error(`Memory with id ${memoryId} not found`);
    }
    const previous = String(existing.payload.data ?? "");
    const sessionFilters = this.sessionScopeFilters(existing.payload);
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
    if (this.entityStore) {
      try {
        await this.removeMemoryFromEntityStore(memoryId, sessionFilters);
      } catch {
        /* non-fatal */
      }
    }
    if (this.graphEnabled && this.graphStore) {
      try {
        await this.graphStore.deleteByMemoryId(memoryId, sessionFilters);
      } catch {
        /* non-fatal */
      }
    }
  }

  // ---------------- entity store ----------------

  private async linkEntities(
    memories: Array<{ id: string; text: string }>,
    filters: SearchFilters,
  ): Promise<void> {
    if (!this.entityStore) return;
    const allEntities = extractEntitiesBatch(memories.map((m) => m.text));
    // Global dedup across all incoming memories.
    const globalEntities = new Map<
      string,
      { entity: Entity; memoryIds: Set<string> }
    >();
    for (let idx = 0; idx < memories.length; idx++) {
      const memId = memories[idx]!.id;
      const ents = allEntities[idx] ?? [];
      for (const e of ents) {
        const key = `${e.type}::${e.text.toLowerCase().trim()}`;
        let entry = globalEntities.get(key);
        if (!entry) {
          entry = { entity: e, memoryIds: new Set() };
          globalEntities.set(key, entry);
        }
        entry.memoryIds.add(memId);
      }
    }
    if (globalEntities.size === 0) return;

    const entityList = Array.from(globalEntities.values());
    const entityTexts = entityList.map((e) => e.entity.text);
    let entityEmbeddings: number[][];
    try {
      entityEmbeddings = await this.embedBatchRetried(entityTexts);
    } catch {
      // Fall back to per-entity embedding so partial failures don't tank
      // the whole batch.
      entityEmbeddings = [];
      for (const t of entityTexts) {
        try {
          entityEmbeddings.push(await this.embedRetried(t));
        } catch {
          entityEmbeddings.push([]);
        }
      }
    }

    const toInsertVectors: number[][] = [];
    const toInsertIds: string[] = [];
    const toInsertPayloads: Record<string, unknown>[] = [];

    for (let i = 0; i < entityList.length; i++) {
      const { entity, memoryIds } = entityList[i]!;
      const vec = entityEmbeddings[i];
      if (!vec || vec.length === 0) continue;
      let matches: Array<{
        id: string;
        score?: number;
        payload: Record<string, unknown>;
      }> = [];
      try {
        matches = await this.entityStore.search(vec, 1, filters);
      } catch {
        matches = [];
      }
      if (matches.length > 0 && (matches[0]!.score ?? 0) >= 0.95) {
        // Update existing entity.
        const m = matches[0]!;
        const linked = new Set<string>(
          Array.isArray(m.payload.linkedMemoryIds)
            ? (m.payload.linkedMemoryIds as string[])
            : [],
        );
        for (const id of memoryIds) linked.add(id);
        const newPayload = {
          ...m.payload,
          linkedMemoryIds: Array.from(linked).sort(),
        };
        try {
          await this.entityStore.update(m.id, vec, newPayload);
        } catch {
          /* non-fatal */
        }
      } else {
        const payload: Record<string, unknown> = {
          data: entity.text,
          entityType: entity.type,
          linkedMemoryIds: Array.from(memoryIds).sort(),
        };
        if (filters.user_id) payload.user_id = filters.user_id;
        if (filters.agent_id) payload.agent_id = filters.agent_id;
        if (filters.run_id) payload.run_id = filters.run_id;
        toInsertVectors.push(vec);
        toInsertIds.push(uuid());
        toInsertPayloads.push(payload);
      }
    }

    if (toInsertVectors.length > 0) {
      try {
        await this.entityStore.insert(
          toInsertVectors,
          toInsertIds,
          toInsertPayloads,
        );
      } catch {
        /* non-fatal */
      }
    }
  }

  private async removeMemoryFromEntityStore(
    memoryId: string,
    filters: SearchFilters,
  ): Promise<void> {
    if (!this.entityStore) return;
    let rows: Array<{ id: string; payload: Record<string, unknown> }> = [];
    try {
      rows = await this.entityStore.list(filters, 10000);
    } catch {
      return;
    }
    for (const row of rows) {
      const linked = Array.isArray(row.payload.linkedMemoryIds)
        ? (row.payload.linkedMemoryIds as string[])
        : [];
      if (!linked.includes(memoryId)) continue;
      const remaining = linked.filter((id) => id !== memoryId);
      if (remaining.length === 0) {
        try {
          await this.entityStore.delete(row.id);
        } catch {
          /* non-fatal */
        }
      } else {
        try {
          const text = String(row.payload.data ?? "");
          if (!text) continue;
          const vec = await this.embedRetried(text);
          await this.entityStore.update(row.id, vec, {
            ...row.payload,
            linkedMemoryIds: remaining,
          });
        } catch {
          /* non-fatal */
        }
      }
    }
  }

  // ---------------- graph memory ----------------

  private async linkGraphTriples(
    memories: Array<{ id: string; text: string }>,
    filters: SearchFilters,
  ): Promise<void> {
    if (!this.graphStore) return;
    const userPrompt = buildTripleExtractionUserPrompt(
      memories.map((m) => m.text),
    );
    let raw: string;
    try {
      raw = await this.llmRetried(
        [
          { role: "system", content: TRIPLE_EXTRACTION_PROMPT },
          { role: "user", content: userPrompt },
        ],
        { responseFormat: "json_object" },
      );
    } catch {
      return;
    }
    let parsed: { triples?: unknown };
    try {
      parsed = JSON.parse(extractJson(raw));
    } catch {
      return;
    }
    if (!parsed || !Array.isArray(parsed.triples)) return;
    const triples: Array<{
      subject: string;
      relation: string;
      object: string;
      filters: SearchFilters;
      memoryId: string;
    }> = [];
    // The LLM doesn't natively know which triple maps to which memory id.
    // We assign each triple to ALL provided memory ids (best-effort linkage).
    for (const item of parsed.triples as unknown[]) {
      if (typeof item !== "object" || item === null) continue;
      const t = item as Record<string, unknown>;
      const subject = typeof t.subject === "string" ? t.subject.trim() : "";
      const relation = typeof t.relation === "string" ? t.relation.trim() : "";
      const object = typeof t.object === "string" ? t.object.trim() : "";
      if (!subject || !relation || !object) continue;
      for (const m of memories) {
        triples.push({ subject, relation, object, filters, memoryId: m.id });
      }
    }
    if (triples.length === 0) return;
    await this.graphStore.addTriples(triples, (text) =>
      this.embedRetried(text),
    );
  }

  // ---------------- search ----------------

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

    // Step 1: Lemmatize + extract entities from the query.
    const queryTokens = lemmatizeForBm25(query);
    const queryEntities = extractEntities(query);

    // Step 2: Embed query.
    const queryEmbedding = await this.embedRetried(query);

    // Step 3: Over-fetch semantic candidates.
    const internalLimit = Math.max(topK * 4, 60);
    const semanticResults = await this.vectorStore.search(
      queryEmbedding,
      internalLimit,
      filters,
    );

    // Step 4: BM25 keyword search if supported.
    const bm25Scores = new Map<string, number>();
    if (
      typeof this.vectorStore.keywordSearch === "function" &&
      queryTokens.length > 0
    ) {
      try {
        const kwResults = await this.vectorStore.keywordSearch(
          queryTokens,
          internalLimit,
          filters,
        );
        const [midpoint, steepness] = getBm25Params(query, queryTokens);
        for (const r of kwResults) {
          const raw = r.score ?? 0;
          if (raw > 0) {
            bm25Scores.set(r.id, normalizeBm25(raw, midpoint, steepness));
          }
        }
      } catch {
        /* non-fatal */
      }
    }

    // Step 5: Entity-store boost.
    const entityBoosts = new Map<string, number>();
    if (queryEntities.length > 0 && this.entityStore) {
      try {
        const dedup = this.dedupEntities(queryEntities, 8);
        for (const ent of dedup) {
          let entityVec: number[];
          try {
            entityVec = await this.embedRetried(ent.text);
          } catch {
            continue;
          }
          let matches: Array<{
            id: string;
            score?: number;
            payload: Record<string, unknown>;
          }> = [];
          try {
            matches = await this.entityStore.search(entityVec, 500, filters);
          } catch {
            continue;
          }
          for (const m of matches) {
            const sim = m.score ?? 0;
            const linked = Array.isArray(m.payload.linkedMemoryIds)
              ? (m.payload.linkedMemoryIds as string[])
              : [];
            const boost = entityBoostFor(sim, linked.length, ENTITY_BOOST_WEIGHT);
            if (boost === 0) continue;
            for (const memId of linked) {
              const prev = entityBoosts.get(memId) ?? 0;
              if (boost > prev) entityBoosts.set(memId, boost);
            }
          }
        }
      } catch {
        /* non-fatal */
      }
    }

    // Step 6: Graph subgraph boost.
    const graphBoosts = new Map<string, number>();
    if (this.graphEnabled && this.graphStore && queryEntities.length > 0) {
      try {
        const dedup = this.dedupEntities(queryEntities, 8);
        const entityEmbeddings: Array<{ text: string; vector: number[] }> = [];
        for (const e of dedup) {
          try {
            entityEmbeddings.push({
              text: e.text,
              vector: await this.embedRetried(e.text),
            });
          } catch {
            /* skip */
          }
        }
        const triples = await this.graphStore.searchByEntities(
          entityEmbeddings,
          { filters, hops: 2, limit: 200 },
        );
        for (const t of triples) {
          if (!t.memoryId) continue;
          const prev = graphBoosts.get(t.memoryId) ?? 0;
          const score = (t.score ?? 0) * 0.4;
          if (score > prev) graphBoosts.set(t.memoryId, score);
        }
      } catch {
        /* non-fatal */
      }
    }

    // Step 7: Score fusion + rank + threshold.
    const candidates: Candidate[] = semanticResults
      .filter((r) => !!r.payload.data)
      .map((r) => ({
        id: r.id,
        score: r.score ?? 0,
        payload: r.payload,
      }));

    const ranked = scoreAndRank(
      candidates,
      bm25Scores,
      entityBoosts,
      graphBoosts,
      threshold,
      topK,
      this.config.scoringWeights,
    );

    return {
      results: ranked.map((r) =>
        this.toMemoryItem({
          id: r.id,
          payload: r.payload,
          score: r.finalScore,
        }),
      ),
    };
  }

  private dedupEntities(entities: Entity[], maxN: number): Entity[] {
    const seen = new Set<string>();
    const out: Entity[] = [];
    for (const e of entities) {
      const key = e.text.trim().toLowerCase();
      if (!key || seen.has(key)) continue;
      seen.add(key);
      out.push(e);
      if (out.length >= maxN) break;
    }
    return out;
  }

  // ---------------- list / mutate / reset ----------------

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
    for (const r of records) await this.applyDelete(r.id);
    return { message: "Memories deleted successfully" };
  }

  async getHistory(memoryId: string): Promise<HistoryRecord[]> {
    await this.initialize();
    return this.history.getHistory(memoryId);
  }

  async reset(): Promise<void> {
    await this.initialize();
    await this.vectorStore.deleteCollection();
    if (this.entityStore) await this.entityStore.deleteCollection();
    if (this.graphStore) await this.graphStore.reset();
    await this.history.reset();
    this.initialized = false;
    await this.initialize();
  }

  async close(): Promise<void> {
    await this.history.close();
  }

  // ---------------- procedural memory ----------------

  /**
   * Compress an agent execution trace into a single procedural memory entry.
   * Uses the PROCEDURAL_MEMORY_PROMPT (verbatim from upstream mem0) to
   * preserve every action's exact output.
   */
  async addProcedural(
    messages: Message[],
    options: ProceduralMemoryOptions = {},
  ): Promise<MemoryItem> {
    await this.initialize();
    if (!Array.isArray(messages) || messages.length === 0) {
      throw new Error("addProcedural requires a non-empty messages array");
    }
    const filters = this.buildFilters({
      userId: options.userId,
      agentId: options.agentId,
      runId: options.runId,
    });
    if (!filters.user_id && !filters.agent_id && !filters.run_id) {
      throw new Error(
        "addProcedural requires at least one of userId, agentId, runId",
      );
    }
    const transcript = messages
      .map((m) => `${m.role}: ${m.content}`)
      .join("\n");
    const summary = await this.llmRetried(
      [
        { role: "system", content: PROCEDURAL_MEMORY_PROMPT },
        { role: "user", content: transcript },
      ],
      { responseFormat: "text" },
    );
    const metadata = {
      ...(options.metadata ?? {}),
      type: "procedural",
    };
    const id = await this.persistMemory(summary, metadata, filters);
    return {
      id,
      memory: summary,
      metadata: { ...metadata, event: "ADD" },
      ...(filters.user_id && { user_id: filters.user_id }),
      ...(filters.agent_id && { agent_id: filters.agent_id }),
      ...(filters.run_id && { run_id: filters.run_id }),
    };
  }
}
