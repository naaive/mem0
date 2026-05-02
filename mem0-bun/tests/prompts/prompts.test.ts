import { describe, expect, test } from "bun:test";
import {
  ADDITIVE_EXTRACTION_PROMPT,
  AGENT_CONTEXT_SUFFIX,
  buildAdditiveExtractionUserPrompt,
  buildTripleExtractionUserPrompt,
  buildUpdateMemoryUserPrompt,
  factRetrievalPrompt,
  PROCEDURAL_MEMORY_PROMPT,
  TRIPLE_EXTRACTION_PROMPT,
  UPDATE_MEMORY_PROMPT,
} from "../../src/prompts";

describe("prompts", () => {
  test("factRetrievalPrompt embeds today's date", () => {
    const p = factRetrievalPrompt("2030-01-02");
    expect(p).toContain("Today's date is 2030-01-02");
    expect(p).toContain("\"facts\": []");
  });

  test("factRetrievalPrompt without arg uses current date", () => {
    const p = factRetrievalPrompt();
    expect(p).toMatch(/Today's date is \d{4}-\d{2}-\d{2}/);
  });

  test("UPDATE_MEMORY_PROMPT mentions all events", () => {
    expect(UPDATE_MEMORY_PROMPT).toContain("ADD");
    expect(UPDATE_MEMORY_PROMPT).toContain("UPDATE");
    expect(UPDATE_MEMORY_PROMPT).toContain("DELETE");
    expect(UPDATE_MEMORY_PROMPT).toContain("NONE");
  });

  test("buildUpdateMemoryUserPrompt includes existing and retrieved", () => {
    const out = buildUpdateMemoryUserPrompt(
      [{ id: "0", text: "old" }],
      ["new fact"],
    );
    expect(out).toContain("old");
    expect(out).toContain("new fact");
    expect(out).not.toContain("Custom instructions");
  });

  test("buildUpdateMemoryUserPrompt includes custom instructions when given", () => {
    const out = buildUpdateMemoryUserPrompt([], [], "be careful");
    expect(out).toContain("Custom instructions");
    expect(out).toContain("be careful");
  });

  test("ADDITIVE_EXTRACTION_PROMPT mentions all events", () => {
    expect(ADDITIVE_EXTRACTION_PROMPT).toContain("ADD");
    expect(ADDITIVE_EXTRACTION_PROMPT).toContain("UPDATE");
    expect(ADDITIVE_EXTRACTION_PROMPT).toContain("DELETE");
    expect(ADDITIVE_EXTRACTION_PROMPT).toContain("NONE");
  });

  test("AGENT_CONTEXT_SUFFIX biases toward agent persona", () => {
    expect(AGENT_CONTEXT_SUFFIX).toContain("scoped to an agent");
  });

  test("buildAdditiveExtractionUserPrompt includes existing + new", () => {
    const out = buildAdditiveExtractionUserPrompt({
      existingMemories: [{ id: "0", text: "old" }],
      newMessages: "new transcript",
    });
    expect(out).toContain("old");
    expect(out).toContain("new transcript");
    expect(out).not.toContain("Custom instructions");
    expect(out).not.toContain("Recent conversation");
  });

  test("buildAdditiveExtractionUserPrompt includes context + custom instructions", () => {
    const out = buildAdditiveExtractionUserPrompt({
      existingMemories: [],
      newMessages: "now",
      lastKMessages: [
        { role: "user", content: "earlier-msg" },
        { role: "assistant", content: "earlier-reply" },
      ],
      customInstructions: "stay focused",
    });
    expect(out).toContain("Recent conversation");
    expect(out).toContain("earlier-msg");
    expect(out).toContain("earlier-reply");
    expect(out).toContain("Custom instructions");
    expect(out).toContain("stay focused");
  });

  test("buildAdditiveExtractionUserPrompt skips empty lastKMessages", () => {
    const out = buildAdditiveExtractionUserPrompt({
      existingMemories: [],
      newMessages: "x",
      lastKMessages: [],
    });
    expect(out).not.toContain("Recent conversation");
  });

  test("TRIPLE_EXTRACTION_PROMPT mentions subject/relation/object", () => {
    expect(TRIPLE_EXTRACTION_PROMPT).toContain("subject");
    expect(TRIPLE_EXTRACTION_PROMPT).toContain("relation");
    expect(TRIPLE_EXTRACTION_PROMPT).toContain("object");
  });

  test("buildTripleExtractionUserPrompt embeds the memory texts", () => {
    const out = buildTripleExtractionUserPrompt(["Alice met Bob"]);
    expect(out).toContain("Alice met Bob");
    expect(out).toContain("triples");
  });

  test("PROCEDURAL_MEMORY_PROMPT preserves verbatim guidance", () => {
    expect(PROCEDURAL_MEMORY_PROMPT).toContain("verbatim");
  });
});
