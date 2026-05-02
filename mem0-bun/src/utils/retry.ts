export interface RetryOptions {
  /** Total number of attempts including the first. Default 3. */
  attempts?: number;
  /** Initial delay in ms. Default 200. */
  baseDelayMs?: number;
  /** Multiplier between attempts. Default 2 (exponential). */
  factor?: number;
  /** Hard cap on delay. Default 5000. */
  maxDelayMs?: number;
  /** Predicate to decide whether an error is retryable. Default true. */
  shouldRetry?: (err: unknown) => boolean;
  /** Sleep implementation; injectable for tests. */
  sleep?: (ms: number) => Promise<void>;
}

const defaultSleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Retry an async function with exponential backoff. The function is invoked
 * up to `attempts` times. If every attempt fails, the last error is rethrown.
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {},
): Promise<T> {
  const attempts = options.attempts ?? 3;
  const baseDelay = options.baseDelayMs ?? 200;
  const factor = options.factor ?? 2;
  const maxDelay = options.maxDelayMs ?? 5000;
  const shouldRetry = options.shouldRetry ?? (() => true);
  const sleep = options.sleep ?? defaultSleep;

  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (err) {
      if (i === attempts - 1 || !shouldRetry(err)) {
        throw err;
      }
      const delay = Math.min(baseDelay * factor ** i, maxDelay);
      await sleep(delay);
    }
  }
  // The loop above always either returns or throws; this throw exists only
  // to satisfy TypeScript's control-flow analysis when `attempts === 0`.
  throw new Error("withRetry: attempts must be >= 1");
}
