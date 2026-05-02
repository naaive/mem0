export interface ScoringWeights {
  /** Weight for cosine semantic score. Default 1.0. */
  semantic?: number;
  /** Weight for normalized BM25 score. Default 0.5. */
  bm25?: number;
  /** Weight for entity-store boost. Default 0.3. */
  entity?: number;
  /** Weight for graph-traversal boost. Default 0.4. */
  graph?: number;
}

const DEFAULT_WEIGHTS: Required<ScoringWeights> = {
  semantic: 1.0,
  bm25: 0.5,
  entity: 0.3,
  graph: 0.4,
};

export interface Candidate {
  id: string;
  /** Cosine similarity (already in [0,1]). */
  score: number;
  payload: Record<string, unknown>;
}

export interface ScoredCandidate extends Candidate {
  /** Final fused score post-weighting (still 0-1ish but not clamped). */
  finalScore: number;
  /** Component breakdown for debugging. */
  components: {
    semantic: number;
    bm25: number;
    entity: number;
    graph: number;
  };
}

/**
 * Spread-attenuated entity boost: an entity linked to many memories
 * contributes less per memory. Matches mem0's reference implementation.
 */
export function entityBoostFor(
  similarity: number,
  numLinked: number,
  baseWeight: number,
): number {
  if (similarity < 0.5) return 0;
  const n = Math.max(numLinked, 1);
  const memoryCountWeight = 1.0 / (1.0 + 0.001 * (n - 1) ** 2);
  return similarity * baseWeight * memoryCountWeight;
}

/** Fuse semantic + BM25 + entity + graph scores and rank. */
export function scoreAndRank(
  candidates: Candidate[],
  bm25Scores: Map<string, number>,
  entityBoosts: Map<string, number>,
  graphBoosts: Map<string, number>,
  threshold: number,
  topK: number,
  weights: ScoringWeights = {},
): ScoredCandidate[] {
  const w: Required<ScoringWeights> = { ...DEFAULT_WEIGHTS, ...weights };
  const scored: ScoredCandidate[] = candidates.map((c) => {
    const bm = bm25Scores.get(c.id) ?? 0;
    const ent = entityBoosts.get(c.id) ?? 0;
    const gph = graphBoosts.get(c.id) ?? 0;
    const finalScore =
      w.semantic * c.score + w.bm25 * bm + w.entity * ent + w.graph * gph;
    return {
      ...c,
      finalScore,
      components: { semantic: c.score, bm25: bm, entity: ent, graph: gph },
    };
  });

  scored.sort((a, b) => b.finalScore - a.finalScore);
  const filtered = scored.filter((s) => s.finalScore >= threshold);
  return filtered.slice(0, topK);
}
