import { describe, expect, test } from "bun:test";
import { MockLLM } from "../../src/llms/mock";

describe("MockLLM", () => {
  test("default responder returns empty string", async () => {
    const llm = new MockLLM();
    const out = await llm.generateResponse([
      { role: "user", content: "hi" },
    ]);
    expect(out).toBe("");
  });

  test("uses custom responder with options", async () => {
    const llm = new MockLLM({
      responder: async (msgs, opts) =>
        `${msgs[0]!.content}|${opts.responseFormat ?? ""}`,
    });
    const out = await llm.generateResponse(
      [{ role: "user", content: "hi" }],
      { responseFormat: "json_object" },
    );
    expect(out).toBe("hi|json_object");
  });

  test("forwards default options object when none provided", async () => {
    const llm = new MockLLM({
      responder: (_msgs, opts) => JSON.stringify(opts),
    });
    const out = await llm.generateResponse([
      { role: "user", content: "x" },
    ]);
    expect(out).toBe("{}");
  });
});
