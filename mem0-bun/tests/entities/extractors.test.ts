import { describe, expect, test } from "bun:test";
import { LocalEntityExtractor } from "../../src/entities/local";
import { LLMEntityExtractor } from "../../src/entities/llm";
import { HybridEntityExtractor } from "../../src/entities/hybrid";
import { MockLLM } from "../../src/llms/mock";
import type { Message } from "../../src/types";
import type { LLMResponseOptions } from "../../src/llms/base";

function llmReturning(reply: string | ((msgs: Message[]) => string)): MockLLM {
  return new MockLLM({
    responder: (msgs: Message[], _opts: LLMResponseOptions) =>
      typeof reply === "function" ? reply(msgs) : reply,
  });
}

describe("LocalEntityExtractor", () => {
  test("delegates to extractEntities + Batch", async () => {
    const e = new LocalEntityExtractor();
    const single = await e.extract("Alice met Bob in Paris.");
    expect(single.length).toBeGreaterThan(0);
    const batch = await e.extractBatch(["Alice", "Bob"]);
    expect(batch.length).toBe(2);
  });
});

describe("LLMEntityExtractor", () => {
  test("returns empty for empty/whitespace input", async () => {
    const e = new LLMEntityExtractor(llmReturning("{}"));
    expect(await e.extract("")).toEqual([]);
    expect(await e.extract("   ")).toEqual([]);
  });

  test("parses valid JSON output", async () => {
    const e = new LLMEntityExtractor(
      llmReturning(
        JSON.stringify({
          entities: [
            { text: "Alice", type: "person" },
            { text: "Paris", type: "place" },
          ],
        }),
      ),
    );
    const out = await e.extract("Alice met Bob in Paris.");
    expect(out.map((x) => `${x.type}:${x.text}`)).toEqual([
      "person:Alice",
      "place:Paris",
    ]);
  });

  test("dedups by lowercase text + type", async () => {
    const e = new LLMEntityExtractor(
      llmReturning(
        JSON.stringify({
          entities: [
            { text: "Alice", type: "person" },
            { text: "alice", type: "person" }, // duplicate
            { text: "Alice", type: "place" }, // different type, kept
          ],
        }),
      ),
    );
    const out = await e.extract("Alice");
    expect(out.length).toBe(2);
    expect(out[0]!.type).toBe("person");
    expect(out[1]!.type).toBe("place");
  });

  test("ignores items with missing text or invalid type", async () => {
    const e = new LLMEntityExtractor(
      llmReturning(
        JSON.stringify({
          entities: [
            { text: "", type: "person" },
            { text: "X", type: "weird-type" },
            "not-an-object",
            null,
            { text: "Alice", type: "person" },
          ],
        }),
      ),
    );
    const out = await e.extract("Alice");
    expect(out.length).toBe(1);
    expect(out[0]!.text).toBe("Alice");
  });

  test("returns empty on invalid JSON", async () => {
    const e = new LLMEntityExtractor(llmReturning("garbage"));
    expect(await e.extract("hello")).toEqual([]);
  });

  test("returns empty when entities array is missing", async () => {
    const e = new LLMEntityExtractor(llmReturning(JSON.stringify({})));
    expect(await e.extract("hello")).toEqual([]);
  });

  test("returns empty on LLM error", async () => {
    const llm = new MockLLM({
      responder: () => {
        throw new Error("LLM down");
      },
    });
    const e = new LLMEntityExtractor(llm);
    expect(await e.extract("hello")).toEqual([]);
  });

  test("clearCache forces a re-call", async () => {
    let calls = 0;
    const llm = new MockLLM({
      responder: () => {
        calls++;
        return JSON.stringify({
          entities: [{ text: "Alice", type: "person" }],
        });
      },
    });
    const e = new LLMEntityExtractor(llm);
    await e.extract("Alice");
    e.clearCache();
    await e.extract("Alice");
    expect(calls).toBe(2);
  });

  test("LRU eviction drops oldest entry past cacheMax", async () => {
    let calls = 0;
    const llm = new MockLLM({
      responder: (msgs) => {
        calls++;
        return JSON.stringify({
          entities: [{ text: msgs[1]!.content, type: "topic" }],
        });
      },
    });
    const e = new LLMEntityExtractor(llm, { cacheMax: 2 });
    await e.extract("a");
    await e.extract("b");
    await e.extract("a"); // refreshes a's LRU position
    await e.extract("c"); // evicts b
    await e.extract("a"); // still cached
    await e.extract("b"); // re-fetched
    expect(calls).toBe(4); // a, b, c, b again
  });

  test("caches results so repeat calls don't re-invoke the LLM", async () => {
    let calls = 0;
    const llm = new MockLLM({
      responder: () => {
        calls++;
        return JSON.stringify({
          entities: [{ text: "Alice", type: "person" }],
        });
      },
    });
    const e = new LLMEntityExtractor(llm);
    await e.extract("Alice");
    await e.extract("Alice");
    expect(calls).toBe(1);
  });

  test("extractBatch loops over inputs", async () => {
    const e = new LLMEntityExtractor(
      llmReturning(
        JSON.stringify({ entities: [{ text: "X", type: "person" }] }),
      ),
    );
    const out = await e.extractBatch(["a", "b"]);
    expect(out.length).toBe(2);
  });
});

describe("HybridEntityExtractor", () => {
  test("returns local result without LLM call when local hits threshold", async () => {
    let llmCalls = 0;
    const llm = new MockLLM({
      responder: () => {
        llmCalls++;
        return JSON.stringify({ entities: [] });
      },
    });
    const h = new HybridEntityExtractor(llm);
    const out = await h.extract("Alice met Bob in Paris.");
    expect(out.length).toBeGreaterThan(0);
    expect(llmCalls).toBe(0);
  });

  test("falls back to LLM when local returns nothing", async () => {
    let llmCalls = 0;
    const llm = new MockLLM({
      responder: () => {
        llmCalls++;
        return JSON.stringify({
          entities: [{ text: "fallback", type: "topic" }],
        });
      },
    });
    const h = new HybridEntityExtractor(llm, {
      llmFallbackThreshold: 1,
      llmForNonLatin: false,
    });
    const out = await h.extract("..."); // trivial input → 0 local entities
    expect(llmCalls).toBe(1);
    expect(out.some((e) => e.text === "fallback")).toBe(true);
  });

  test("falls back to LLM for non-Latin script even when local found something", async () => {
    let llmCalls = 0;
    const llm = new MockLLM({
      responder: () => {
        llmCalls++;
        return JSON.stringify({
          entities: [{ text: "北京", type: "place" }],
        });
      },
    });
    const h = new HybridEntityExtractor(llm); // default: llmForNonLatin = true
    const out = await h.extract("我去了北京。");
    expect(llmCalls).toBe(1);
    // Output is a union of local CJK chunks + LLM entities (deduped).
    expect(out.some((e) => e.text === "北京" && e.type === "place")).toBe(true);
  });

  test("merges local + llm results without duplicates", async () => {
    const llm = new MockLLM({
      responder: () =>
        JSON.stringify({
          entities: [
            { text: "Alice", type: "person" }, // duplicate of local
            { text: "extra", type: "topic" },
          ],
        }),
    });
    const h = new HybridEntityExtractor(llm, {
      llmFallbackThreshold: 100, // force fallback
      llmForNonLatin: false,
    });
    const out = await h.extract("Alice met Bob in Paris.");
    const persons = out.filter((e) => e.type === "person").map((e) => e.text);
    // Alice appears once despite both extractors emitting it.
    expect(persons.filter((p) => /^alice$/i.test(p)).length).toBe(1);
    expect(out.some((e) => e.type === "topic" && e.text === "extra")).toBe(true);
  });

  test("returns local-only when LLM returns empty", async () => {
    const llm = new MockLLM({
      responder: () => JSON.stringify({ entities: [] }),
    });
    const h = new HybridEntityExtractor(llm, {
      llmFallbackThreshold: 100,
      llmForNonLatin: false,
    });
    const out = await h.extract("Alice met Bob in Paris.");
    expect(out.length).toBeGreaterThan(0);
  });

  test("respects llmForNonLatin=false (English-only mode)", async () => {
    let llmCalls = 0;
    const llm = new MockLLM({
      responder: () => {
        llmCalls++;
        return JSON.stringify({ entities: [] });
      },
    });
    const h = new HybridEntityExtractor(llm, {
      llmForNonLatin: false,
    });
    await h.extract("Alice met Bob in Paris.");
    expect(llmCalls).toBe(0);
  });

  test("extractBatch loops over inputs", async () => {
    const llm = new MockLLM({
      responder: () => JSON.stringify({ entities: [] }),
    });
    const h = new HybridEntityExtractor(llm, { llmForNonLatin: false });
    const out = await h.extractBatch(["a", "b"]);
    expect(out.length).toBe(2);
  });
});
