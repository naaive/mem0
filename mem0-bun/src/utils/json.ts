/**
 * Extract a JSON object from a model response. Strips ```json fences and
 * trailing prose. Returns the raw substring; caller is responsible for parsing.
 */
export function extractJson(input: string): string {
  if (!input) return "";
  let text = input.trim();

  // Strip markdown code fences ```json ... ``` or ``` ... ```
  const fenceMatch = text.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  if (fenceMatch && fenceMatch[1]) {
    text = fenceMatch[1].trim();
  }

  // Find first balanced JSON object/array
  const startIdx = text.search(/[{\[]/);
  if (startIdx === -1) return text;

  const opening = text[startIdx];
  const closing = opening === "{" ? "}" : "]";
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = startIdx; i < text.length; i++) {
    const ch = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (ch === "\\") {
      escape = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (ch === opening) depth++;
    else if (ch === closing) {
      depth--;
      if (depth === 0) {
        return text.slice(startIdx, i + 1);
      }
    }
  }
  return text.slice(startIdx);
}

export function safeJsonParse<T = unknown>(input: string, fallback: T): T {
  try {
    return JSON.parse(input) as T;
  } catch {
    return fallback;
  }
}
