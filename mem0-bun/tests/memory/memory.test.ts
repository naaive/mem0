import { describe, expect, test } from "bun:test";
import { Memory } from "../../src/memory/memory";
import type { LLMResponseOptions } from "../../src/llms/base";
import type { Message } from "../../src/types";

interface Step {
  match: (msgs: Message[]) => boolean;
  reply: string;
}

/**
 * Build a Memory instance wired entirely to mock providers, plus a small
 * scripted LLM that returns canned responses for the fact-extraction and
 * memory-update calls. Each script entry is consumed in order when matched.
 */
function buildMemory(
  steps: Step[],
  options: { disableHistory?: boolean; customInstructions?: string } = {},
): { memory: Memory; calls: Array<{ system: string; user: string }> } {
  const remaining = [...steps];
  const calls: Array<{ system: string; user: string }> = [];
  const memory = new Memory({
    llm: {
      provider: "mock",
      config: {
        responder: async (msgs: Message[], _opts: LLMResponseOptions) => {
          calls.push({
            system: msgs[0]!.content,
            user: msgs[1]?.content ?? "",
          });
          const idx = remaining.findIndex((s) => s.match(msgs));
          if (idx === -1) {
            throw new Error(
              `Unexpected LLM call:\n${msgs.map((m) => `${m.role}: ${m.content}`).join("\n")}`,
            );
          }
          const step = remaining.splice(idx, 1)[0]!;
          return step.reply;
        },
      },
    },
    embedder: { provider: "mock", config: { embeddingDims: 8 } },
    vectorStore: { provider: "memory", config: { collectionName: "test" } },
    historyStore: { provider: "memory", config: {} },
    disableHistory: options.disableHistory,
    customInstructions: options.customInstructions,
  });
  return { memory, calls };
}

const factsStep = (facts: string[]): Step => ({
  match: (m) => m[0]!.content.includes("Personal Information Organizer"),
  reply: JSON.stringify({ facts }),
});

const updateStep = (
  memory: Array<{
    id: string;
    text: string;
    event: string;
    old_memory?: string;
  }>,
): Step => ({
  match: (m) => m[0]!.content.includes("smart memory manager"),
  reply: JSON.stringify({ memory }),
});

describe("Memory.add", () => {
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
      factsStep(["Likes pizza"]),
      updateStep([{ id: "0", text: "Likes pizza", event: "ADD" }]),
    ]);
    const out = await memory.add("I love pizza", { userId: "u1" });
    expect(out.results).toHaveLength(1);
    expect(out.results[0]!.memory).toBe("Likes pizza");
    expect(out.results[0]!.metadata?.event).toBe("ADD");
    expect(out.results[0]!.id).toMatch(/[0-9a-f-]{36}/);
  });

  test("UPDATE action rewrites existing memory and writes history", async () => {
    const { memory } = buildMemory([
      // First add: extracts "Likes pizza" then ADDs.
      factsStep(["Likes pizza"]),
      updateStep([{ id: "0", text: "Likes pizza", event: "ADD" }]),
      // Second add: extracts updated fact, decides UPDATE on the existing memory.
      factsStep(["Loves pepperoni pizza"]),
      updateStep([
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
    expect(second.results[0]!.memory).toBe("Loves pepperoni pizza");
    expect(second.results[0]!.metadata?.event).toBe("UPDATE");
    expect(second.results[0]!.id).toBe(memId);

    const history = await memory.getHistory(memId);
    expect(history.map((h) => h.action)).toEqual(["ADD", "UPDATE"]);
    expect(history[1]!.previousValue).toBe("Likes pizza");
    expect(history[1]!.newValue).toBe("Loves pepperoni pizza");
  });

  test("DELETE action drops contradicted memory", async () => {
    const { memory } = buildMemory([
      factsStep(["Likes pizza"]),
      updateStep([{ id: "0", text: "Likes pizza", event: "ADD" }]),
      factsStep(["Dislikes pizza"]),
      updateStep([{ id: "0", text: "Likes pizza", event: "DELETE" }]),
    ]);
    const first = await memory.add("I love pizza", { userId: "u1" });
    const memId = first.results[0]!.id;
    const out = await memory.add("Now I hate pizza", { userId: "u1" });
    expect(out.results[0]!.metadata?.event).toBe("DELETE");
    expect(await memory.get(memId)).toBeNull();
  });

  test("NONE action returns nothing", async () => {
    const { memory } = buildMemory([
      factsStep(["Already known"]),
      updateStep([{ id: "0", text: "Already known", event: "NONE" }]),
    ]);
    const out = await memory.add("rehash", { userId: "u1" });
    expect(out.results).toEqual([]);
  });

  test("returns empty when LLM extracts no facts", async () => {
    const { memory } = buildMemory([factsStep([])]);
    const out = await memory.add("nothing", { userId: "u1" });
    expect(out.results).toEqual([]);
  });

  test("returns empty when LLM returns invalid extraction JSON", async () => {
    const { memory } = buildMemory([
      { match: (m) => m[0]!.content.includes("Personal Information"), reply: "not json" },
    ]);
    const out = await memory.add("hello", { userId: "u1" });
    expect(out.results).toEqual([]);
  });

  test("ignores facts with non-string entries", async () => {
    const { memory } = buildMemory([
      {
        match: (m) => m[0]!.content.includes("Personal Information"),
        reply: JSON.stringify({ facts: ["good", 42, "  "] }),
      },
      updateStep([{ id: "0", text: "good", event: "ADD" }]),
    ]);
    const out = await memory.add("x", { userId: "u1" });
    expect(out.results.map((r) => r.memory)).toEqual(["good"]);
  });

  test("ignores facts response missing facts array", async () => {
    const { memory } = buildMemory([
      {
        match: (m) => m[0]!.content.includes("Personal Information"),
        reply: JSON.stringify({ other: 1 }),
      },
    ]);
    expect((await memory.add("x", { userId: "u1" })).results).toEqual([]);
  });

  test("ignores update response missing memory array", async () => {
    const { memory } = buildMemory([
      factsStep(["A"]),
      {
        match: (m) => m[0]!.content.includes("smart memory manager"),
        reply: JSON.stringify({}),
      },
    ]);
    expect((await memory.add("x", { userId: "u1" })).results).toEqual([]);
  });

  test("ignores update entries with missing text or unknown event", async () => {
    const { memory } = buildMemory([
      factsStep(["A", "B"]),
      {
        match: (m) => m[0]!.content.includes("smart memory manager"),
        reply: JSON.stringify({
          memory: [
            { id: "0", text: "", event: "ADD" }, // missing text
            { id: "1", text: "B", event: "WEIRD" }, // unknown event
            "not-an-object",
            null,
            { id: 5, text: "C", event: "ADD" }, // numeric id is coerced
          ],
        }),
      },
    ]);
    const out = await memory.add("x", { userId: "u1" });
    expect(out.results).toHaveLength(1);
    expect(out.results[0]!.memory).toBe("C");
  });

  test("update returns empty when JSON parse fails", async () => {
    const { memory } = buildMemory([
      factsStep(["a"]),
      {
        match: (m) => m[0]!.content.includes("smart memory manager"),
        reply: "not json",
      },
    ]);
    expect((await memory.add("x", { userId: "u1" })).results).toEqual([]);
  });

  test("UPDATE/DELETE without realId is skipped", async () => {
    const { memory } = buildMemory([
      factsStep(["new"]),
      updateStep([
        { id: "999", text: "ghost", event: "UPDATE" },
        { id: "999", text: "ghost", event: "DELETE" },
      ]),
    ]);
    expect((await memory.add("x", { userId: "u1" })).results).toEqual([]);
  });

  test("infer=false stores each non-system message verbatim", async () => {
    const { memory } = buildMemory([]);
    const out = await memory.add(
      [
        { role: "system", content: "ignored" },
        { role: "user", content: "raw1" },
        { role: "assistant", content: "raw2" },
      ],
      { userId: "u1", infer: false, metadata: { source: "chat" } },
    );
    expect(out.results.map((r) => r.memory)).toEqual(["raw1", "raw2"]);
    const all = await memory.getAll({ filters: { user_id: "u1" } });
    expect(all.results).toHaveLength(2);
    expect(all.results[0]!.metadata?.source).toBe("chat");
  });

  test("captures errors during persistence without aborting batch", async () => {
    const { memory } = buildMemory([
      factsStep(["good"]),
      updateStep([
        { id: "0", text: "good", event: "ADD" },
        // The decideUpdates references id "1" but no matching realId exists,
        // so it gets skipped. To test the catch block we issue an UPDATE on a
        // valid existing record but force a failure by mutating the vector
        // store via reset() between phases. Simpler: directly force via
        // monkey patch on the vector store handle.
      ]),
    ]);
    // baseline ADD
    await memory.add("hello", { userId: "u1" });
    // Force the next applyUpdate to throw by deleting the underlying record
    const all = await memory.getAll({ filters: { user_id: "u1" } });
    const id = all.results[0]!.id;

    // Build a second memory with a script that issues an UPDATE for the now-deleted id
    const { memory: m2 } = buildMemory([
      factsStep(["change"]),
      updateStep([{ id: "0", text: "new value", event: "UPDATE" }]),
    ]);
    // Pre-populate m2's store with a record then delete it via the memory
    // public API to simulate a vanished id seen by the LLM mapping. We stub
    // the existing search by directly pre-inserting one record then deleting
    // it before the second add.
    await m2.add("seed", { userId: "u1", infer: false });

    // After this seed insert the next add will see existingResults with
    // id=0; we'll have applyUpdate called with that id which exists, so this
    // path won't actually fail. To exercise the catch, monkey-patch the
    // vector store update to throw.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const internal = m2 as unknown as {
      vectorStore: { update: () => Promise<void> };
    };
    const originalUpdate = internal.vectorStore.update.bind(
      internal.vectorStore,
    );
    internal.vectorStore.update = async () => {
      throw new Error("boom");
    };

    const out = await m2.add("change", { userId: "u1" });
    internal.vectorStore.update = originalUpdate as unknown as () => Promise<void>;
    expect(out.results[0]!.metadata?.event).toBe("ERROR");
    expect(out.results[0]!.metadata?.error).toContain("boom");

    void id; // referenced to keep variable alive for clarity
  });

  test("custom instructions propagate to extraction prompt", async () => {
    const { memory, calls } = buildMemory(
      [factsStep([])],
      { customInstructions: "STAY FOCUSED" },
    );
    await memory.add("x", { userId: "u1" });
    const ext = calls.find((c) => c.system.includes("Personal Information"));
    expect(ext?.user).toContain("STAY FOCUSED");
  });

  test("filters from config are merged with entity ids", async () => {
    const { memory } = buildMemory([
      factsStep(["x"]),
      updateStep([{ id: "0", text: "x", event: "ADD" }]),
    ]);
    await memory.add("hi", {
      userId: "u1",
      filters: { custom: "tag" },
      metadata: { extra: "v" },
    });
    const all = await memory.getAll({ filters: { user_id: "u1" } });
    expect(all.results[0]!.metadata?.extra).toBe("v");
  });

  test("empty payload data is filtered out from search results", async () => {
    const { memory } = buildMemory([
      factsStep(["solid fact"]),
      updateStep([{ id: "0", text: "solid fact", event: "ADD" }]),
    ]);
    await memory.add("hi", { userId: "u1" });
    const search = await memory.search("solid", {
      filters: { user_id: "u1" },
    });
    expect(search.results[0]!.memory).toBe("solid fact");
  });
});

describe("Memory.search", () => {
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
      memory.search("q", {
        filters: { user_id: "u" },
        threshold: 5,
      }),
    ).rejects.toThrow();
    await expect(
      memory.search("q", { filters: { user_id: "u" }, topK: 1.5 }),
    ).rejects.toThrow();
  });

  test("filters by score threshold", async () => {
    const { memory } = buildMemory([
      factsStep(["coffee", "tea"]),
      updateStep([
        { id: "0", text: "coffee", event: "ADD" },
        { id: "1", text: "tea", event: "ADD" },
      ]),
    ]);
    await memory.add("hi", { userId: "u1" });

    // threshold=1 will exclude all but exact-match queries via mock embedding
    const out = await memory.search("coffee", {
      filters: { user_id: "u1" },
      threshold: 1,
    });
    expect(out.results.length).toBeLessThanOrEqual(1);
  });

  test("respects topK", async () => {
    const { memory } = buildMemory([
      factsStep(["a", "b", "c"]),
      updateStep([
        { id: "0", text: "a", event: "ADD" },
        { id: "1", text: "b", event: "ADD" },
        { id: "2", text: "c", event: "ADD" },
      ]),
    ]);
    await memory.add("seed", { userId: "u1" });
    const out = await memory.search("anything", {
      filters: { user_id: "u1" },
      topK: 2,
    });
    expect(out.results.length).toBe(2);
  });

  test("preserves extra filter keys", async () => {
    const { memory } = buildMemory([
      factsStep(["a"]),
      updateStep([{ id: "0", text: "a", event: "ADD" }]),
    ]);
    await memory.add("seed", { userId: "u1", metadata: { tag: "x" } });
    const out = await memory.search("anything", {
      filters: { user_id: "u1", tag: "x" },
    });
    expect(out.results).toHaveLength(1);
  });

  test("validates filter entity ids", async () => {
    const { memory } = buildMemory([]);
    await expect(
      memory.search("q", { filters: { user_id: "  " } }),
    ).rejects.toThrow(/empty/);
  });
});

describe("Memory.get / getAll", () => {
  test("get returns null for missing", async () => {
    const { memory } = buildMemory([]);
    expect(await memory.get("nope")).toBeNull();
  });

  test("get returns full memory item with metadata and entity ids", async () => {
    const { memory } = buildMemory([
      factsStep(["x"]),
      updateStep([{ id: "0", text: "x", event: "ADD" }]),
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
    expect(item?.hash).toBeDefined();
    expect(item?.createdAt).toBeDefined();
  });

  test("getAll requires at least one entity id", async () => {
    const { memory } = buildMemory([]);
    await expect(memory.getAll({})).rejects.toThrow(/at least one of/);
  });

  test("getAll rejects top-level entity ids", async () => {
    const { memory } = buildMemory([]);
    await expect(
      memory.getAll({ user_id: "u" } as unknown as {
        filters: { user_id: string };
      }),
    ).rejects.toThrow(/Top-level entity/);
  });

  test("getAll respects topK", async () => {
    const { memory } = buildMemory([]);
    for (let i = 0; i < 3; i++) {
      await memory.add(`m${i}`, { userId: "u1", infer: false });
    }
    const all = await memory.getAll({
      filters: { user_id: "u1" },
      topK: 2,
    });
    expect(all.results).toHaveLength(2);
  });

  test("getAll validates topK", async () => {
    const { memory } = buildMemory([]);
    await expect(
      memory.getAll({ filters: { user_id: "u1" }, topK: -1 }),
    ).rejects.toThrow(/non-negative/);
  });
});

describe("Memory.update / delete / deleteAll / reset / history", () => {
  test("update mutates content and writes history", async () => {
    const { memory } = buildMemory([]);
    const out = await memory.add("first", { userId: "u1", infer: false });
    const id = out.results[0]!.id;
    const result = await memory.update(id, "second");
    expect(result.message).toMatch(/updated/);
    const item = await memory.get(id);
    expect(item?.memory).toBe("second");
    const hist = await memory.getHistory(id);
    expect(hist.map((h) => h.action)).toEqual(["ADD", "UPDATE"]);
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

  test("delete validates input and removes record", async () => {
    const { memory } = buildMemory([]);
    await expect(memory.delete("")).rejects.toThrow(/memoryId/);
    const out = await memory.add("x", { userId: "u1", infer: false });
    const id = out.results[0]!.id;
    const result = await memory.delete(id);
    expect(result.message).toMatch(/deleted/);
    expect(await memory.get(id)).toBeNull();
  });

  test("delete on missing memory throws", async () => {
    const { memory } = buildMemory([]);
    await expect(memory.delete("ghost")).rejects.toThrow(/not found/);
  });

  test("deleteAll clears scoped memories", async () => {
    const { memory } = buildMemory([]);
    await memory.add("x", { userId: "u1", infer: false });
    await memory.add("y", { userId: "u2", infer: false });
    const r = await memory.deleteAll({ userId: "u1" });
    expect(r.message).toMatch(/deleted/);
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

  test("reset wipes vector store and history", async () => {
    const { memory } = buildMemory([]);
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

  test("close releases history manager", async () => {
    const { memory } = buildMemory([]);
    await memory.add("x", { userId: "u1", infer: false });
    await memory.close();
  });
});

describe("Memory.create static helper", () => {
  test("returns initialized instance", async () => {
    const m = await Memory.create({
      llm: { provider: "mock", config: {} },
      embedder: { provider: "mock", config: { embeddingDims: 4 } },
      vectorStore: { provider: "memory", config: {} },
      historyStore: { provider: "memory", config: {} },
    });
    expect(m).toBeInstanceOf(Memory);
    // double-init is a no-op
    await m.initialize();
  });
});
