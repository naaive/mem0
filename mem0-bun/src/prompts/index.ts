/**
 * Prompts for fact extraction, memory updates, triple extraction, and
 * procedural-memory summarization. Mirrors the upstream mem0 prompts but
 * tightened for Bun + structured JSON outputs.
 */

export function factRetrievalPrompt(
  today: string = new Date().toISOString().slice(0, 10),
): string {
  return `You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences. Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts.

Types of Information to Remember:
1. Personal Preferences (likes, dislikes, food, products, activities, entertainment).
2. Important Personal Details (names, relationships, dates).
3. Plans and Intentions (events, trips, goals).
4. Activity and Service Preferences (dining, travel, hobbies).
5. Health and Wellness (dietary restrictions, fitness routines).
6. Professional Details (job titles, work habits, career goals).
7. Miscellaneous (favorite books, movies, brands).

Few-shot examples:
Input: Hi.
Output: {"facts": []}

Input: Hi, I am looking for a restaurant in San Francisco.
Output: {"facts": ["Looking for a restaurant in San Francisco"]}

Input: Yesterday, I had a meeting with John at 3pm. We discussed the new project.
Output: {"facts": ["Had a meeting with John at 3pm", "Discussed the new project"]}

Input: Hi, my name is John. I am a software engineer.
Output: {"facts": ["Name is John", "Is a Software engineer"]}

Return JSON of the form {"facts": ["..."]}.

Rules:
- Today's date is ${today}.
- Do not echo the few-shot examples.
- Extract from user/assistant messages only; ignore system messages.
- If nothing is relevant return {"facts": []}.
- Detect the language of the user input and record facts in the same language.`;
}

export const UPDATE_MEMORY_PROMPT = `You are a smart memory manager controlling a long-term memory store.

You can perform four operations against the existing memory:
- ADD: a new fact not already present.
- UPDATE: an existing fact whose information has changed; keep its original id.
- DELETE: an existing fact contradicted by the new facts.
- NONE: no change (already present or irrelevant).

Compare the retrieved facts with the existing memory list, then return STRICT JSON in this format:
{
  "memory": [
    { "id": "<existing or new>", "text": "<fact>", "event": "ADD|UPDATE|DELETE|NONE", "old_memory": "<only if UPDATE>" }
  ]
}

Guidelines:
- For ADD events, generate a new id (the caller will replace it with a UUID).
- For UPDATE events, return the SAME id from the existing memory.
- For DELETE events, return the SAME id from the existing memory.
- For NONE events, copy the existing entry unchanged.
- Do not invent new facts. Only operate on the inputs provided.
- Output ONLY JSON. No prose.`;

export function buildUpdateMemoryUserPrompt(
  existing: Array<{ id: string; text: string }>,
  retrieved: string[],
  customInstructions?: string,
): string {
  const parts: string[] = [];
  if (customInstructions) {
    parts.push(`Custom instructions:\n${customInstructions}\n`);
  }
  parts.push(`Existing memory:\n${JSON.stringify(existing, null, 2)}\n`);
  parts.push(`Retrieved facts:\n${JSON.stringify(retrieved, null, 2)}\n`);
  parts.push(
    `Return the new memory state as JSON of the form {"memory":[{"id":"...","text":"...","event":"...","old_memory":"..."}]}.`,
  );
  return parts.join("\n");
}

/**
 * Single-call additive extraction prompt. Replaces the older two-call
 * (extract → diff) flow with one LLM round. Outputs only NEW or CHANGED
 * facts and avoids re-emitting facts already in the existing memory list.
 */
export const ADDITIVE_EXTRACTION_PROMPT = `You are mem0's long-term memory manager. You receive:
- The existing memory list (each entry has a stable integer id).
- The most recent K turns of conversation (for context only).
- New incoming messages between the user and the assistant.

Your job: emit ONLY new or changed memories that should be stored. Do not echo memories that are already present in the existing list. Detect contradictions and emit DELETE actions for invalidated memories. Detect updates and emit UPDATE actions on the existing id.

For ADD entries you may include "attributed_to" identifying the speaker (e.g. "user", "assistant", or a named agent).

Output STRICT JSON ONLY in the form:
{
  "memory": [
    { "id": "<existing-id|new>", "text": "<fact>", "event": "ADD|UPDATE|DELETE|NONE", "old_memory": "<only for UPDATE>", "attributed_to": "<optional>" }
  ]
}

Rules:
- Use language matching the conversation.
- "id" for ADD is the string "new"; for UPDATE/DELETE/NONE it must match an existing id.
- Do not invent facts. Only operate on the inputs provided.
- Skip system messages when extracting.
- If nothing should change, return {"memory": []}.`;

export const AGENT_CONTEXT_SUFFIX = `

Note: this conversation is scoped to an agent (no user_id present). Bias extraction toward facts about the assistant's persona, capabilities, plans, and stated preferences rather than the user.`;

export function buildAdditiveExtractionUserPrompt(opts: {
  existingMemories: Array<{ id: string; text: string }>;
  newMessages: string;
  lastKMessages?: Array<{ role: string; content: string }>;
  customInstructions?: string;
}): string {
  const parts: string[] = [];
  if (opts.customInstructions) {
    parts.push(`Custom instructions:\n${opts.customInstructions}`);
  }
  parts.push(
    `Existing memory:\n${JSON.stringify(opts.existingMemories, null, 2)}`,
  );
  if (opts.lastKMessages && opts.lastKMessages.length > 0) {
    parts.push(
      `Recent conversation (context only):\n${opts.lastKMessages
        .map((m) => `${m.role}: ${m.content}`)
        .join("\n")}`,
    );
  }
  parts.push(`New messages to extract from:\n${opts.newMessages}`);
  parts.push(
    `Return JSON of the form {"memory":[{"id":"...","text":"...","event":"...","old_memory":"...","attributed_to":"..."}]}.`,
  );
  return parts.join("\n\n");
}

/**
 * Triple extraction prompt for graph memory. Asks the LLM to convert each
 * memory text into a list of (subject, relation, object) triples, where
 * subjects/objects are concrete entities (people, places, things, events).
 */
export const TRIPLE_EXTRACTION_PROMPT = `You convert memory facts into knowledge-graph triples.

For each input fact, extract one or more triples capturing the relations between entities. Triples must be of the form:
{ "subject": "<entity>", "relation": "<verb_or_predicate>", "object": "<entity_or_value>" }

Guidelines:
- Use concise entity names (no articles, no possessives).
- Relations are short snake_case predicates like "lives_in", "likes", "works_at", "born_on", "has_role".
- Drop facts that have no clear (subject, relation, object) shape.
- Return STRICT JSON: {"triples": [...]}.`;

export function buildTripleExtractionUserPrompt(memoryTexts: string[]): string {
  return `Extract triples for these facts:\n${JSON.stringify(memoryTexts, null, 2)}\n\nReturn JSON of the form {"triples":[{"subject":"...","relation":"...","object":"..."}]}.`;
}

/**
 * Procedural memory: compress an agent execution trace into a single
 * structured summary that preserves every action's exact output.
 * Adapted verbatim from upstream mem0 prompts.py.
 */
export const PROCEDURAL_MEMORY_PROMPT = `You are a memory summarization system that records and preserves the complete interaction history between a human and an AI agent. Your task is to produce a comprehensive summary of the agent's output history that contains every detail necessary for the agent to continue the task without ambiguity. **Every output produced by the agent must be recorded verbatim as part of the summary.**

Structure the summary as:
1. Overview — Task Objective + Progress Status (% complete, completed milestones).
2. Sequential Agent Actions (numbered) — for each step include:
   a. Agent Action (precise description, parameters, target elements).
   b. Action Result (mandatory, unmodified, verbatim).
   c. Embedded metadata: key findings, navigation history, errors and challenges, current context.

Rules:
- Preserve every output verbatim.
- Chronological order.
- Include exact data: URLs, indices, error messages, JSON payloads.
- Output ONLY the structured summary; no preamble.`;
