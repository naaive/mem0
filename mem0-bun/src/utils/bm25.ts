/**
 * Standalone BM25 helpers.
 *
 * The convention follows mem0's reference TS implementation:
 *  - `getBm25Params` selects a sigmoid (midpoint, steepness) pair used to
 *    map raw BM25 scores into [0, 1] so they can be fused with cosine.
 *  - `normalizeBm25` applies that sigmoid.
 *  - `scoreBm25` computes the canonical Robertson/Spärck Jones BM25 score
 *    for a tokenized query against a tokenized document.
 */

export interface CorpusStats {
  /** Total number of documents in the corpus. */
  docCount: number;
  /** Average document length in tokens. */
  avgDocLen: number;
  /** Document frequency per term (number of docs containing the term). */
  df: Map<string, number>;
}

export interface Bm25Options {
  k1?: number;
  b?: number;
}

const DEFAULT_K1 = 1.5;
const DEFAULT_B = 0.75;

/**
 * Build corpus stats from a list of pre-tokenized (lemmatized) documents.
 * Each entry is a token array.
 */
export function buildCorpusStats(documents: string[][]): CorpusStats {
  const df = new Map<string, number>();
  let totalLen = 0;
  for (const doc of documents) {
    totalLen += doc.length;
    const seen = new Set<string>();
    for (const tok of doc) {
      if (seen.has(tok)) continue;
      seen.add(tok);
      df.set(tok, (df.get(tok) ?? 0) + 1);
    }
  }
  return {
    docCount: documents.length,
    avgDocLen: documents.length === 0 ? 0 : totalLen / documents.length,
    df,
  };
}

function termFreq(tokens: string[]): Map<string, number> {
  const tf = new Map<string, number>();
  for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
  return tf;
}

/**
 * Compute the BM25 score for a single document given query tokens and
 * pre-computed corpus stats.
 */
export function scoreBm25(
  queryTokens: string[],
  docTokens: string[],
  stats: CorpusStats,
  options: Bm25Options = {},
): number {
  const k1 = options.k1 ?? DEFAULT_K1;
  const b = options.b ?? DEFAULT_B;
  if (stats.docCount === 0 || docTokens.length === 0) return 0;
  const tf = termFreq(docTokens);
  const docLen = docTokens.length;
  const lenNorm = stats.avgDocLen === 0 ? 1 : docLen / stats.avgDocLen;
  let score = 0;
  for (const term of queryTokens) {
    const f = tf.get(term);
    if (!f) continue;
    const docFreq = stats.df.get(term) ?? 0;
    // Robertson-Sparck Jones IDF with +0.5 smoothing
    const idf = Math.log(
      1 + (stats.docCount - docFreq + 0.5) / (docFreq + 0.5),
    );
    const num = f * (k1 + 1);
    const denom = f + k1 * (1 - b + b * lenNorm);
    score += idf * (num / denom);
  }
  return score;
}

/**
 * Sigmoid normalization onto [0, 1] using a midpoint and steepness.
 * Matches mem0's reference implementation so fusion weights stay portable.
 */
export function normalizeBm25(
  rawScore: number,
  midpoint: number,
  steepness: number,
): number {
  if (steepness === 0) return rawScore >= midpoint ? 1 : 0;
  return 1 / (1 + Math.exp(-steepness * (rawScore - midpoint)));
}

/**
 * Heuristic to pick (midpoint, steepness) given the query. Longer / more
 * informative queries deserve a higher midpoint so scores are not
 * overinflated; this mirrors the upstream tuning.
 */
export function getBm25Params(
  query: string,
  queryLemmatized: string[],
): [midpoint: number, steepness: number] {
  const queryLen = queryLemmatized.length;
  if (queryLen <= 1) return [1.5, 1.5];
  if (queryLen <= 3) return [3, 1.2];
  if (queryLen <= 6) return [5, 1.0];
  // Longer queries (heuristic check on raw text length too)
  const charLen = query.length;
  if (charLen > 100) return [8, 0.6];
  return [6, 0.8];
}
