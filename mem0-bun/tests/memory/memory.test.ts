import { describe, expect, test } from "bun:test";
import { Memory } from "../../src/memory/memory";
import type { LLMResponseOptions } from "../../src/llms/base";
import type { MemoryConfig, Message } from "../../src/types";

interface Step {
  match: (msgs: Message[]) => boolean;
  reply: string | ((msgs: Message[]) => string);
}

interface Build {
  memory: Memory;
  calls: Array<{ system: string; user: string }>;
}

/**
 * Build a Memory instance wired entirely to mock providers, with a scripted
 * mock LLM. Each script entry is consumed in declaration order when matched.
 */
function buildMemory(
  steps: Step[],
  options: Partial<MemoryConfig> & { embeddingDims?: number } = {},
): Build {
  const remaining = [...steps];
  const calls: Array<{ system: string; user: string }> = [];
  const memory = new Memory({
    llm: {
      provider: "mock",
      config: {
        responder: async (msgs: Message[], _opts: LLMResponseOptions) => {
          calls.push({
            system: msgs[0]?.content ?? "",
            user: msgs[1]?.content ?? "",
          });
          const idx = remaining.findIndex((s) => s.match(msgs));
          if (idx === -1) {
            throw new Error(
              `Unexpected LLM call:\n${msgs.map((m) => `${m.role}: ${m.content.slice(0, 80)}`).join("\n")}`,
            );
          }
          const step = remaining.splice(idx, 1)[0]!;
          return typeof step.reply === "function"
            ? step.reply(msgs)
            : step.reply;
        },
      },
    },
    embedder: {
      provider: "mock",
      config: { embeddingDims: options.embeddingDims ?? 16 },
    },
    vectorStore: { provider: "memory", config: { collectionName: "test" } },
    historyStore: { provider: "memory", config: {} },
    // Disable retries so unexpected-call errors surface immediately.
    retry: { attempts: 1, baseDelayMs: 0 },
    ...options,
  });
  return { memory, calls };
}

const additiveStep = (
  memory: Array<{
    id: string;
    text: string;
    event: string;
    old_memory?: string;
    attributed_to?: string;
  }>,
): Step => ({
  match: (m) => m[0]!.content.includes("long-term memory manager"),
  reply: JSON.stringify({ memory }),
});

const tripleStep = (
  triples: Array<{ subject: string; relation: string; object: string }>,
): Step => ({
  match: (m) => m[0]!.content.includes("knowledge-graph triples"),
  reply: JSON.stringify({ triples }),
});

const proceduralStep = (text: string): Step => ({
  match: (m) => m[0]!.content.includes("memory summarization system"),
  reply: text,
});

describe("Memory.add (V3 additive flow)", () => {
  test("requires at least one entity id", async () => {
    const { memory } = buildMemory([]);
    await expect(memory.add("hello")).rejects.toThrow(/userId/);
  });

  test("rejects null/undefined messages", async () => {
    const { memory } = buildMemory([]);
    await expect(
      memory.add(null as unknown as string, { userId: "u" }),
    ).rejects.toThrow(/required/);
  });

  test("ADD action persists a new memory", async () => {
    const { memory } = buildMemory([
      additiveStep([{ id: "new", text: "Likes pizza", event: "ADD" }]),
    ]);
    const out = await memory.add("I love pizza", { userId: "u1" });
    expect(out.results).toHaveLength(1);
    expect(out.results[0]!.memory).toBe("Likes pizza");
    expect(out.results[0]!.metadata?.event).toBe("ADD");
  });

  test("UPDATE rewrites existing memory and writes history", async () => {
    const { memory } = buildMemory([
      additiveStep([{ id: "new", text: "Likes pizza", event: "ADD" }]),
      additiveStep([
        {
          id: "0",
          text: "Loves pepperoni pizza",
          event: "UPDATE",
          old_memory: "Likes pizza",
        },
      ]),
    ]);
    const first = await memory.add("I love pizza", { userId: "u1" });
    const memId = first.results[0]!.id;
    const second = await memory.add("Actually I love pepperoni pizza", {
      userId: "u1",
    });
    expect(second.results[0]!.metadata?.event).toBe("UPDATE");
    expect(second.results[0]!.id).toBe(memId);
    const history = await memory.getHistory(memId);
    expect(history.map((h) => h.action)).toEqual(["ADD", "UPDATE"]);
    expect(history[1]!.previousValue).toBe("Likes pizza");
  });

  test("DELETE drops contradicted memory", async () => {
    const { memory } = buildMemory([
      additiveStep([{ id: "new", text: "Likes pizza", event: "ADD" }]),
      additiveStep([{ id: "0", text: "Likes pizza", event: "DELETE" }]),
    ]);
    const first = await memory.add("I love pizza", { userId: "u1" });
    const id = first.results[0]!.id;
    const out = await memory.add("Now I hate pizza", { userId: "u1" });
    expect(out.results[0]!.metadata?.event).toBe("DELETE");
    expect(await memory.get(id)).toBeNull();
  });

  test("NONE returns nothing", async () => {
    const { memory } = buildMemory([
      additiveStep([{ id: "0", text: "Already known", event: "NONE" }]),
    ]);
    const out = await memory.add("rehash", { userId: "u1" });
    expect(out.results).toEqual([]);
  });

  test("returns empty when LLM returns invalid JSON", async () => {
    const { memory } = buildMemory([
      {
        match: (m) => m[0]!.content.includes("long-term memory manager"),
        reply: "not json",
      },
    ]);
    const out = await memory.add("hi", { userId: "u1" });
    expect(out.results).toEqual([]);
  });

  test("returns empty when LLM throws (after retries exhausted)", async () => {
    const memory = new Memory({
      llm: {
        provider: "mock",
        config: {
          responder: () => {
            throw new Error("LLM down");
          },
        },
      },
      embedder: { provider: "mock", config: { embeddingDims: 16 } },
      vectorStore: { provider: "memory", config: { collectionName: "t" } },
      historyStore: { provider: "memory", config: {} },
      retry: { attempts: 1, baseDelayMs: 0 },
    });
    const out = await memory.add("hi", { userId: "u1" });
    expect(out.results).toEqual([]);
  });

  test("ignores entries with non-string text or unknown event", async () => {
    const { memory } = buildMemory([
      {
        match: (m) => m[0]!.content.includes("long-term memory manager"),
        reply: JSON.stringify({
          memory: [
            { id: "new", text: "", event: "ADD" }, // empty text
            { id: "new", text: "valid", event: "WEIRD" }, // unknown event
            "not-an-object",
            null,
            { id: 5, text: "second", event: "ADD" }, // numeric id coerced
          ],
        }),
      },
    ]);
    const out = await memory.add("hi", { userId: "u1" });
    expect(out.results).toHaveLength(1);
    expect(out.results[0]!.memory).toBe("second");
  });

  test("dedups extracted facts by content hash", async () => {
    const { memory } = buildMemory([
      additiveStep([
        { id: "new", text: "Likes pizza", event: "ADD" },
        { id: "new", text: "Likes pizza", event: "ADD" }, // duplicate
      ]),
    ]);
    const out = await memory.add("hi", { userId: "u1" });
    expect(out.results).toHaveLength(1);
  });

  test("ignores response missing memory array", async () => {
    const { memory } = buildMemory([
      {
        match: (m) => m[0]!.content.includes("long-term memory manager"),
        reply: JSON.stringify({}),
      },
    ]);
    expect((await memory.add("x", { userId: "u1" })).results).toEqual([]);
  });

  test("UPDATE/DELETE with no realId is silently dropped", async () => {
    const { memory } = buildMemory([
      additiveStep([
        { id: "999", text: "phantom", event: "UPDATE" },
        { id: "999", text: "phantom", event: "DELETE" },
      ]),
    ]);
    expect((await memory.add("x", { userId: "u1" })).results).toEqual([]);
  });

  test("infer=false stores each non-system message verbatim", async () => {
    const { memory } = buildMemory([]);
    const out = await memory.add(
      [
        { role: "system", content: "ignore" },
        { role: "user", content: "raw1" },
        { role: "assistant", content: "raw2" },
      ],
      { userId: "u1", infer: false, metadata: { source: "chat" } },
    );
    expect(out.results.map((r) => r.memory)).toEqual(["raw1", "raw2"]);
  });

  test("agent-scoped (no user_id) appends agent-context suffix to system prompt", async () => {
    const { memory, calls } = buildMemory([
      additiveStep([{ id: "new", text: "Persona is curious", event: "ADD" }]),
    ]);
    await memory.add("hi", { agentId: "agent1" });
    expect(calls[0]!.system).toContain("scoped to an agent");
  });

  test("attributed_to is preserved on output and persisted", async () => {
    const { memory } = buildMemory([
      additiveStep([
        {
          id: "new",
          text: "Likes pizza",
          event: "ADD",
          attributed_to: "user",
        },
      ]),
    ]);
    const out = await memory.add("hi", { userId: "u1" });
    expect(out.results[0]!.metadata?.attributedTo).toBe("user");
  });

  test("captures errors during persistence without aborting batch", async () => {
    const { memory } = buildMemory([
      additiveStep([{ id: "new", text: "ok", event: "ADD" }]),
    ]);
    await memory.add("seed", { userId: "u1" });

    const { memory: m2 } = buildMemory([
      additiveStep([{ id: "0", text: "new value", event: "UPDATE" }]),
    ]);
    await m2.add("seed", { userId: "u1", infer: false });

    const internal = m2 as unknown as {
      vectorStore: { update: () => Promise<void> };
    };
    const original = internal.vectorStore.update.bind(internal.vectorStore);
    internal.vectorStore.update = async () => {
      throw new Error("boom");
    };
    const out = await m2.add("change", { userId: "u1" });
    internal.vectorStore.update = original as unknown as () => Promise<void>;
    expect(out.results[0]!.metadata?.event).toBe("ERROR");
    expect(out.results[0]!.metadata?.error).toContain("boom");
  });

  test("custom instructions propagate to extraction prompt", async () => {
    const { memory, calls } = buildMemory(
      [additiveStep([])],
      { customInstructions: "STAY FOCUSED" },
    );
    await memory.add("hi", { userId: "u1" });
    expect(calls[0]!.user).toContain("STAY FOCUSED");
  });
});

describe("Memory.search (hybrid retrieval)", () => {
  test("requires at least one entity id in filters", async () => {
    const { memory } = buildMemory([]);
    await expect(memory.search("q", { filters: {} })).rejects.toThrow(
      /at least one of/,
    );
  });

  test("rejects top-level entity ids", async () => {
    const { memory } = buildMemory([]);
    await expect(
      memory.search("q", {
        user_id: "u",
      } as unknown as { filters: { user_id: string } }),
    ).rejects.toThrow(/Top-level entity/);
  });

  test("validates threshold and topK", async () => {
    const { memory } = buildMemory([]);
    await expect(
      memory.search("q", { filters: { user_id: "u" }, threshold: 5 }),
    ).rejects.toThrow();
    await expect(
      memory.search("q", { filters: { user_id: "u" }, topK: 1.5 }),
    ).rejects.toThrow();
  });

  test("BM25 keyword search lifts a memory above unrelated semantics", async () => {
    const { memory } = buildMemory([
      additiveStep([
        { id: "new", text: "User loves pepperoni pizza", event: "ADD" },
        { id: "new", text: "User dislikes broccoli", event: "ADD" },
      ]),
    ]);
    await memory.add("seed", { userId: "u1" });
    const out = await memory.search("pepperoni pizza", {
      filters: { user_id: "u1" },
    });
    expect(out.results).toHaveLength(2);
    expect(out.results[0]!.memory).toContain("pepperoni");
  });

  test("entity boost lifts memories linked to query entities", async () => {
    const { memory } = buildMemory([
      additiveStep([
        { id: "new", text: "Alice met Bob in Paris yesterday", event: "ADD" },
        { id: "new", text: "Random unrelated fact", event: "ADD" },
      ]),
    ]);
    await memory.add("seed", { userId: "u1" });
    const out = await memory.search("Tell me about Alice and Paris", {
      filters: { user_id: "u1" },
    });
    expect(out.results[0]!.memory).toContain("Alice");
  });

  test("preserves extra filter keys", async () => {
    const { memory } = buildMemory([
      additiveStep([{ id: "new", text: "tagged fact", event: "ADD" }]),
    ]);
    await memory.add("seed", { userId: "u1", metadata: { tag: "x" } });
    const out = await memory.search("tagged", {
      filters: { user_id: "u1", tag: "x" },
    });
    expect(out.results.length).toBeGreaterThan(0);
  });

  test("validates filter entity ids", async () => {
    const { memory } = buildMemory([]);
    await expect(
      memory.search("q", { filters: { user_id: "  " } }),
    ).rejects.toThrow(/empty/);
  });

  test("respects topK", async () => {
    const { memory } = buildMemory([
      additiveStep([
        { id: "new", text: "first", event: "ADD" },
        { id: "new", text: "second", event: "ADD" },
        { id: "new", text: "third", event: "ADD" },
      ]),
    ]);
    await memory.add("seed", { userId: "u1" });
    const out = await memory.search("anything related to facts", {
      filters: { user_id: "u1" },
      topK: 2,
    });
    expect(out.results.length).toBeLessThanOrEqual(2);
  });

  test("threshold filters out low-score candidates", async () => {
    const { memory } = buildMemory([
      additiveStep([{ id: "new", text: "hello", event: "ADD" }]),
    ]);
    await memory.add("seed", { userId: "u1" });
    const out = await memory.search("totally unrelated query", {
      filters: { user_id: "u1" },
      threshold: 0.99,
    });
    expect(out.results).toEqual([]);
  });

  test("custom scoring weights apply", async () => {
    const { memory } = buildMemory([
      additiveStep([
        { id: "new", text: "Alice met Bob", event: "ADD" },
      ]),
    ], { scoringWeights: { semantic: 0, bm25: 1, entity: 0, graph: 0 } });
    await memory.add("seed", { userId: "u1" });
    const out = await memory.search("Alice", {
      filters: { user_id: "u1" },
    });
    expect(out.results.length).toBe(1);
  });
});

describe("Memory.get / getAll", () => {
  test("get returns null for missing", async () => {
    const { memory } = buildMemory([]);
    expect(await memory.get("nope")).toBeNull();
  });

  test("get returns full memory item with metadata + entity ids", async () => {
    const { memory } = buildMemory([
      additiveStep([{ id: "new", text: "x", event: "ADD" }]),
    ]);
    const out = await memory.add("hi", {
      userId: "u1",
      agentId: "a1",
      runId: "r1",
      metadata: { color: "blue" },
    });
    const id = out.results[0]!.id;
    const item = await memory.get(id);
    expect(item?.user_id).toBe("u1");
    expect(item?.agent_id).toBe("a1");
    expect(item?.run_id).toBe("r1");
    expect(item?.metadata?.color).toBe("blue");
  });

  test("getAll requires entity scope", async () => {
    const { memory } = buildMemory([]);
    await expect(memory.getAll({})).rejects.toThrow(/at least one/);
  });

  test("getAll rejects top-level entity ids", async () => {
    const { memory } = buildMemory([]);
    await expect(
      memory.getAll({ user_id: "u" } as unknown as {
        filters: { user_id: string };
      }),
    ).rejects.toThrow(/Top-level entity/);
  });

  test("getAll respects topK and validates", async () => {
    const { memory } = buildMemory([]);
    for (let i = 0; i < 3; i++) {
      await memory.add(`m${i}`, { userId: "u1", infer: false });
    }
    expect(
      (await memory.getAll({ filters: { user_id: "u1" }, topK: 2 })).results,
    ).toHaveLength(2);
    await expect(
      memory.getAll({ filters: { user_id: "u1" }, topK: -1 }),
    ).rejects.toThrow(/non-negative/);
  });
});

describe("Memory.update / delete / deleteAll / reset", () => {
  test("update mutates content and writes history", async () => {
    const { memory } = buildMemory([]);
    const out = await memory.add("first", { userId: "u1", infer: false });
    const id = out.results[0]!.id;
    const r = await memory.update(id, "second");
    expect(r.message).toMatch(/updated/);
    expect((await memory.get(id))?.memory).toBe("second");
    expect((await memory.getHistory(id)).map((h) => h.action)).toEqual([
      "ADD",
      "UPDATE",
    ]);
  });

  test("update validates inputs", async () => {
    const { memory } = buildMemory([]);
    await expect(memory.update("", "x")).rejects.toThrow(/memoryId/);
    await expect(memory.update("id", "")).rejects.toThrow(/text/);
  });

  test("update on missing memory throws", async () => {
    const { memory } = buildMemory([]);
    await expect(memory.update("nope", "x")).rejects.toThrow(/not found/);
  });

  test("delete removes record and validates input", async () => {
    const { memory } = buildMemory([]);
    await expect(memory.delete("")).rejects.toThrow(/memoryId/);
    const out = await memory.add("x", { userId: "u1", infer: false });
    const r = await memory.delete(out.results[0]!.id);
    expect(r.message).toMatch(/deleted/);
  });

  test("delete on missing memory throws", async () => {
    const { memory } = buildMemory([]);
    await expect(memory.delete("ghost")).rejects.toThrow(/not found/);
  });

  test("deleteAll clears scoped memories", async () => {
    const { memory } = buildMemory([]);
    await memory.add("x", { userId: "u1", infer: false });
    await memory.add("y", { userId: "u2", infer: false });
    await memory.deleteAll({ userId: "u1" });
    expect(
      (await memory.getAll({ filters: { user_id: "u1" } })).results,
    ).toHaveLength(0);
    expect(
      (await memory.getAll({ filters: { user_id: "u2" } })).results,
    ).toHaveLength(1);
  });

  test("deleteAll requires entity scope", async () => {
    const { memory } = buildMemory([]);
    await expect(memory.deleteAll({})).rejects.toThrow(/at least one/);
  });

  test("reset wipes everything (including entity store + graph)", async () => {
    const { memory } = buildMemory([], {
      graphStore: { provider: "memory", config: {} },
      enableGraph: true,
    });
    await memory.add("x", { userId: "u1", infer: false });
    await memory.reset();
    expect(
      (await memory.getAll({ filters: { user_id: "u1" } })).results,
    ).toHaveLength(0);
  });

  test("disableHistory keeps memories without history rows", async () => {
    const { memory } = buildMemory([], { disableHistory: true });
    const out = await memory.add("x", { userId: "u1", infer: false });
    const id = out.results[0]!.id;
    expect(await memory.getHistory(id)).toEqual([]);
    await memory.update(id, "y");
    expect(await memory.getHistory(id)).toEqual([]);
    await memory.delete(id);
    expect(await memory.getHistory(id)).toEqual([]);
  });

  test("close releases history backend", async () => {
    const { memory } = buildMemory([]);
    await memory.add("x", { userId: "u1", infer: false });
    await memory.close();
  });
});

describe("Memory entityExtractor selection", () => {
  test("mode=llm constructs LLMEntityExtractor", () => {
    const { memory } = buildMemory([], {
      entityExtractor: { mode: "llm" },
    });
    expect(memory).toBeDefined();
  });
  test("mode=hybrid constructs HybridEntityExtractor", () => {
    const { memory } = buildMemory([], {
      entityExtractor: { mode: "hybrid", llmFallbackThreshold: 0 },
    });
    expect(memory).toBeDefined();
  });
  test("default falls through to local extractor", () => {
    const { memory } = buildMemory([], {});
    expect(memory).toBeDefined();
  });
});

describe("Memory.create static helper", () => {
  test("returns initialized instance, double init is no-op", async () => {
    const m = await Memory.create({
      llm: { provider: "mock", config: {} },
      embedder: { provider: "mock", config: { embeddingDims: 4 } },
      vectorStore: { provider: "memory", config: {} },
      historyStore: { provider: "memory", config: {} },
    });
    await m.initialize();
    expect(m).toBeInstanceOf(Memory);
  });
});

describe("Memory graph memory (mem0+)", () => {
  test("graph triples are extracted on add and boost related queries", async () => {
    const { memory } = buildMemory(
      [
        additiveStep([
          { id: "new", text: "Alice lives in Paris", event: "ADD" },
        ]),
        tripleStep([
          { subject: "Alice", relation: "lives_in", object: "Paris" },
        ]),
        additiveStep([
          { id: "new", text: "Bob enjoys cycling", event: "ADD" },
        ]),
        tripleStep([
          { subject: "Bob", relation: "enjoys", object: "cycling" },
        ]),
      ],
      {
        graphStore: { provider: "memory", config: {} },
        enableGraph: true,
      },
    );
    await memory.add("Alice lives in Paris", { userId: "u1" });
    await memory.add("Bob enjoys cycling", { userId: "u1" });
    const out = await memory.search("Tell me about Alice's home city", {
      filters: { user_id: "u1" },
    });
    expect(out.results[0]!.memory).toContain("Alice");
  });

  test("triple extraction handles invalid JSON", async () => {
    const { memory } = buildMemory(
      [
        additiveStep([{ id: "new", text: "Alice runs", event: "ADD" }]),
        {
          match: (m) => m[0]!.content.includes("knowledge-graph triples"),
          reply: "garbage",
        },
      ],
      {
        graphStore: { provider: "memory", config: {} },
        enableGraph: true,
      },
    );
    const out = await memory.add("Alice runs", { userId: "u1" });
    expect(out.results).toHaveLength(1);
  });

  test("triple extraction handles missing triples array", async () => {
    const { memory } = buildMemory(
      [
        additiveStep([{ id: "new", text: "X", event: "ADD" }]),
        {
          match: (m) => m[0]!.content.includes("knowledge-graph triples"),
          reply: JSON.stringify({}),
        },
      ],
      {
        graphStore: { provider: "memory", config: {} },
        enableGraph: true,
      },
    );
    await memory.add("X", { userId: "u1" });
  });

  test("triple extraction skips entries with empty fields", async () => {
    const { memory } = buildMemory(
      [
        additiveStep([{ id: "new", text: "X", event: "ADD" }]),
        {
          match: (m) => m[0]!.content.includes("knowledge-graph triples"),
          reply: JSON.stringify({
            triples: [
              { subject: "", relation: "r", object: "o" },
              { subject: "S", relation: "", object: "o" },
              { subject: "S", relation: "r", object: "" },
              "not-an-object",
              null,
              { subject: "Alice", relation: "knows", object: "Bob" },
            ],
          }),
        },
      ],
      {
        graphStore: { provider: "memory", config: {} },
        enableGraph: true,
      },
    );
    await memory.add("X", { userId: "u1" });
  });

  test("LLM throwing on triple extraction is non-fatal", async () => {
    const { memory } = buildMemory(
      [additiveStep([{ id: "new", text: "X", event: "ADD" }])],
      {
        graphStore: { provider: "memory", config: {} },
        enableGraph: true,
      },
    );
    // The triple-step matcher above is unscripted, so the LLM call throws.
    // The pipeline should still return the ADDed memory.
    const out = await memory.add("X", { userId: "u1" });
    expect(out.results).toHaveLength(1);
  });

  test("graph triples are removed on memory delete", async () => {
    const { memory } = buildMemory(
      [
        additiveStep([{ id: "new", text: "Alice loves Paris", event: "ADD" }]),
        tripleStep([
          { subject: "Alice", relation: "loves", object: "Paris" },
        ]),
      ],
      {
        graphStore: { provider: "memory", config: {} },
        enableGraph: true,
      },
    );
    const out = await memory.add("Alice loves Paris", { userId: "u1" });
    const id = out.results[0]!.id;
    await memory.delete(id);
    // Graph cleanup must be silent and idempotent.
  });
});

describe("Memory procedural memory", () => {
  test("addProcedural summarizes agent trace", async () => {
    const { memory } = buildMemory([proceduralStep("## Summary\nstep 1: ok")]);
    const item = await memory.addProcedural(
      [
        { role: "agent", content: "Open URL https://example.com" },
        { role: "tool", content: "200 OK" },
      ],
      { agentId: "a1" },
    );
    expect(item.memory).toContain("Summary");
    expect(item.metadata?.type).toBe("procedural");
  });

  test("addProcedural validates inputs", async () => {
    const { memory } = buildMemory([]);
    await expect(memory.addProcedural([])).rejects.toThrow(/non-empty/);
    await expect(
      memory.addProcedural([{ role: "x", content: "y" }]),
    ).rejects.toThrow(/at least one/);
  });
});
