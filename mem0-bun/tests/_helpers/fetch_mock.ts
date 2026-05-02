/**
 * Tiny fetch stub helper used by provider tests.
 * Replaces the global fetch with a queue or matcher and restores it on dispose.
 */

export type MockResponse = {
  status?: number;
  body?: unknown;
  bodyText?: string;
};

export type RequestRecord = {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: string | undefined;
};

export interface FetchMock {
  calls: RequestRecord[];
  dispose: () => void;
}

function buildResponse(spec: MockResponse): Response {
  const status = spec.status ?? 200;
  const body =
    spec.bodyText !== undefined
      ? spec.bodyText
      : spec.body !== undefined
        ? JSON.stringify(spec.body)
        : "";
  return new Response(body, {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

type FetchInput = Parameters<typeof fetch>[0];

export function mockFetchSequence(responses: MockResponse[]): FetchMock {
  const original = globalThis.fetch;
  const calls: RequestRecord[] = [];
  let index = 0;
  globalThis.fetch = (async (
    input: FetchInput,
    init?: RequestInit,
  ): Promise<Response> => {
    const url = typeof input === "string" ? input : input.toString();
    const headers: Record<string, string> = {};
    if (init?.headers) {
      const h = new Headers(init.headers);
      h.forEach((v, k) => {
        headers[k] = v;
      });
    }
    calls.push({
      url,
      method: init?.method ?? "GET",
      headers,
      body: typeof init?.body === "string" ? init.body : undefined,
    });
    if (index >= responses.length) {
      throw new Error(
        `mockFetchSequence exhausted at request #${index + 1} for ${url}`,
      );
    }
    const spec = responses[index++]!;
    return buildResponse(spec);
  }) as typeof fetch;
  return {
    calls,
    dispose: () => {
      globalThis.fetch = original;
    },
  };
}

export type FetchHandler = (
  url: string,
  init: RequestInit | undefined,
) => MockResponse | Promise<MockResponse>;

export function mockFetchHandler(handler: FetchHandler): FetchMock {
  const original = globalThis.fetch;
  const calls: RequestRecord[] = [];
  globalThis.fetch = (async (
    input: FetchInput,
    init?: RequestInit,
  ): Promise<Response> => {
    const url = typeof input === "string" ? input : input.toString();
    const headers: Record<string, string> = {};
    if (init?.headers) {
      const h = new Headers(init.headers);
      h.forEach((v, k) => {
        headers[k] = v;
      });
    }
    calls.push({
      url,
      method: init?.method ?? "GET",
      headers,
      body: typeof init?.body === "string" ? init.body : undefined,
    });
    const spec = await handler(url, init);
    return buildResponse(spec);
  }) as typeof fetch;
  return {
    calls,
    dispose: () => {
      globalThis.fetch = original;
    },
  };
}
