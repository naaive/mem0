import natural from "natural";

const stemmer: { stem: (word: string) => string } = natural.PorterStemmer;
const Tokenizer = natural.WordTokenizer as unknown as {
  new (): { tokenize: (text: string) => string[] };
};
const tokenizer = new Tokenizer();

// English stopwords — a curated subset matching what natural.stopwords ships
// (we hardcode it so behavior is deterministic across versions).
const STOPWORDS = new Set<string>([
  "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
  "any", "are", "aren", "as", "at", "be", "because", "been", "before", "being",
  "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn",
  "did", "didn", "do", "does", "doesn", "doing", "don", "down", "during",
  "each", "few", "for", "from", "further", "had", "hadn", "has", "hasn", "have",
  "haven", "having", "he", "her", "here", "hers", "herself", "him", "himself",
  "his", "how", "i", "if", "in", "into", "is", "isn", "it", "its", "itself",
  "just", "ll", "m", "me", "might", "more", "most", "must", "my", "myself",
  "needn", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only",
  "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s",
  "same", "shan", "she", "should", "so", "some", "such", "t", "than", "that",
  "the", "their", "theirs", "them", "themselves", "then", "there", "these",
  "they", "this", "those", "through", "to", "too", "under", "until", "up",
  "ve", "very", "was", "wasn", "we", "were", "weren", "what", "when", "where",
  "which", "while", "who", "whom", "why", "will", "with", "won", "would",
  "wouldn", "y", "you", "your", "yours", "yourself", "yourselves",
]);

/** Tokenize a free-text string into lowercase alphanumeric words. */
export function tokenize(text: string): string[] {
  if (!text) return [];
  return tokenizer.tokenize(text).map((t: string) => t.toLowerCase());
}

/** Return the Porter stem of a single token. */
export function stem(token: string): string {
  return stemmer.stem(token);
}

/**
 * Tokenize, drop stopwords, and stem each token. The output preserves order
 * and duplicates so downstream BM25 can compute term frequencies.
 */
export function lemmatizeForBm25(text: string): string[] {
  return tokenize(text)
    .filter((tok) => tok.length > 1 && !STOPWORDS.has(tok))
    .map((tok) => stem(tok));
}

/** Read-only view of the stopword set (for tests and docs). */
export function stopwordsSet(): ReadonlySet<string> {
  return STOPWORDS;
}
