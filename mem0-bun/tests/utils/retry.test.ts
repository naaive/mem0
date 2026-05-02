import { describe, expect, test } from "bun:test";
import { withRetry } from "../../src/utils/retry";

describe("withRetry", () => {
  test("returns immediately on first success", async () => {
    let calls = 0;
    const out = await withRetry(async () => {
      calls++;
      return "ok";
    });
    expect(out).toBe("ok");
    expect(calls).toBe(1);
  });

  test("retries until success", async () => {
    let calls = 0;
    const sleeps: number[] = [];
    const out = await withRetry(
      async () => {
        calls++;
        if (calls < 3) throw new Error("fail");
        return "ok";
      },
      {
        attempts: 5,
        baseDelayMs: 10,
        factor: 2,
        sleep: async (ms) => {
          sleeps.push(ms);
        },
      },
    );
    expect(out).toBe("ok");
    expect(calls).toBe(3);
    expect(sleeps).toEqual([10, 20]);
  });

  test("rethrows last error after exhausting attempts", async () => {
    let calls = 0;
    await expect(
      withRetry(
        async () => {
          calls++;
          throw new Error(`fail-${calls}`);
        },
        { attempts: 3, baseDelayMs: 0, sleep: async () => {} },
      ),
    ).rejects.toThrow("fail-3");
    expect(calls).toBe(3);
  });

  test("respects shouldRetry predicate", async () => {
    let calls = 0;
    await expect(
      withRetry(
        async () => {
          calls++;
          throw new Error("permanent");
        },
        {
          attempts: 5,
          baseDelayMs: 0,
          sleep: async () => {},
          shouldRetry: () => false,
        },
      ),
    ).rejects.toThrow();
    expect(calls).toBe(1);
  });

  test("clamps delay to maxDelayMs", async () => {
    const sleeps: number[] = [];
    let calls = 0;
    await withRetry(
      async () => {
        calls++;
        if (calls < 4) throw new Error("x");
        return "ok";
      },
      {
        attempts: 5,
        baseDelayMs: 100,
        factor: 10,
        maxDelayMs: 200,
        sleep: async (ms) => {
          sleeps.push(ms);
        },
      },
    );
    // Computed: 100, 1000→clamp 200, 10000→clamp 200
    expect(sleeps).toEqual([100, 200, 200]);
  });

  test("default sleep is real (no injected sleep) — exercises real timer", async () => {
    let calls = 0;
    const out = await withRetry(
      async () => {
        calls++;
        if (calls < 2) throw new Error("retry me");
        return "ok";
      },
      { attempts: 3, baseDelayMs: 1 },
    );
    expect(out).toBe("ok");
    expect(calls).toBe(2);
  });

  test("attempts=0 throws the guard error", async () => {
    await expect(
      withRetry(async () => "never", { attempts: 0 }),
    ).rejects.toThrow(/attempts must be >= 1/);
  });
});
