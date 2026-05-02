import { describe, expect, test } from "bun:test";
import {
  buildUpdateMemoryUserPrompt,
  factRetrievalPrompt,
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
});
