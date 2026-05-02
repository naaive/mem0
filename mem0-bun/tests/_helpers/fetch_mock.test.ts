import { describe, expect, test } from "bun:test";
import { mockFetchHandler, mockFetchSequence } from "./fetch_mock";

describe("fetch_mock helper", () => {
  test("mockFetchSequence throws when exhausted", async () => {
    const m = mockFetchSequence([{ body: { ok: 1 } }]);
    try {
      await fetch("https://x.test/a");
      await expect(fetch("https://x.test/b")).rejects.toThrow(/exhausted/);
    } finally {
      m.dispose();
    }
  });

  test("mockFetchSequence supports raw bodyText and default GET", async () => {
    const m = mockFetchSequence([{ status: 201, bodyText: "hi" }]);
    try {
      const res = await fetch("https://x.test/raw");
      expect(res.status).toBe(201);
      expect(await res.text()).toBe("hi");
      expect(m.calls[0]!.method).toBe("GET");
      expect(m.calls[0]!.body).toBeUndefined();
    } finally {
      m.dispose();
    }
  });

  test("mockFetchSequence handles requests without a body init", async () => {
    const m = mockFetchSequence([{}]);
    try {
      await fetch("https://x.test/empty");
      expect(m.calls[0]!.url).toBe("https://x.test/empty");
    } finally {
      m.dispose();
    }
  });

  test("mockFetchSequence captures non-string bodies as undefined", async () => {
    const m = mockFetchSequence([{}]);
    try {
      const blob = new Blob(["hi"]);
      await fetch("https://x.test/blob", { method: "POST", body: blob });
      expect(m.calls[0]!.body).toBeUndefined();
    } finally {
      m.dispose();
    }
  });

  test("mockFetchHandler wires arbitrary URL routes", async () => {
    const m = mockFetchHandler(async (url) => {
      if (url.endsWith("/a")) return { body: { route: "a" } };
      return { status: 404, body: { route: "miss" } };
    });
    try {
      const a = await (await fetch("https://x.test/a")).json();
      const b = await (await fetch("https://x.test/b")).json();
      expect(a).toEqual({ route: "a" });
      expect(b).toEqual({ route: "miss" });
      expect(m.calls).toHaveLength(2);
    } finally {
      m.dispose();
    }
  });

  test("mockFetchHandler captures request body and headers", async () => {
    const m = mockFetchHandler(() => ({ body: {} }));
    try {
      await fetch("https://x.test/ok", {
        method: "POST",
        headers: { "X-Test": "1" },
        body: "abc",
      });
      expect(m.calls[0]!.headers["x-test"]).toBe("1");
      expect(m.calls[0]!.body).toBe("abc");
    } finally {
      m.dispose();
    }
  });

  test("mockFetchHandler accepts URL objects and missing init", async () => {
    const m = mockFetchHandler(() => ({ body: {} }));
    try {
      await fetch(new URL("https://x.test/u"));
      expect(m.calls[0]!.url).toBe("https://x.test/u");
    } finally {
      m.dispose();
    }
  });
});
