/**
 * OpenClaw Memory V2 Plugin
 *
 * Phase 1: Structured memory with typed entries, importance scores, tags (keyword search)
 * Phase 2: Local embeddings via @xenova/transformers for semantic search
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { Type } from "@sinclair/typebox";
import * as fs from "node:fs";
import { homedir } from "node:os";
import * as path from "node:path";

// ============================================================================
// Types
// ============================================================================

type MemoryType = "learning" | "decision" | "interaction" | "event" | "insight";

type MemoryRelationType =
  | "caused"
  | "caused_by"
  | "related"
  | "supersedes"
  | "contradicts"
  | "elaborates";

type MemoryRelation = {
  id: string;
  type: MemoryRelationType;
};

type V2MemoryEntry = {
  id: string;
  timestamp: string;
  date: string;
  type: MemoryType;
  importance: number;
  content: string;
  file: string;
  line: number;
  tags: string[];
  context?: string;
  filePath?: string;
  embedding?: number[] | string; // number[] = legacy v2, string = base64 v3
  relations?: MemoryRelation[];
  accessCount?: number;
  lastAccessed?: string;
};

type V2MemoryIndex = {
  version: string;
  lastUpdated: string;
  memories: V2MemoryEntry[];
  embeddingModel?: string;
};

type EmbeddingProvider = "none" | "local";

type MemoryV2Config = {
  indexPath: string;
  embedding: {
    provider: EmbeddingProvider;
    modelName?: string;
  };
};

// ============================================================================
// Config
// ============================================================================

function parseConfig(value: unknown): MemoryV2Config {
  const defaultPath = path.join(
    homedir(),
    ".openclaw",
    "workspace",
    "memory",
    "index",
    "memory-index.jsonl",
  );

  const defaults: MemoryV2Config = {
    indexPath: defaultPath,
    embedding: {
      provider: "none",
      modelName: "Xenova/all-MiniLM-L6-v2",
    },
  };

  if (!value || typeof value !== "object") {
    return defaults;
  }

  const cfg = value as Record<string, unknown>;
  const indexPath = typeof cfg.indexPath === "string" ? cfg.indexPath : defaultPath;

  const resolved = indexPath.startsWith("~") ? path.join(homedir(), indexPath.slice(1)) : indexPath;

  const embCfg = cfg.embedding as Record<string, unknown> | undefined;
  const embedding: MemoryV2Config["embedding"] = { ...defaults.embedding };

  if (embCfg) {
    if (embCfg.provider === "local") {
      embedding.provider = embCfg.provider;
    }
    if (typeof embCfg.modelName === "string") {
      embedding.modelName = embCfg.modelName;
    }
  }

  return { indexPath: resolved, embedding };
}

// ============================================================================
// Embeddings (Phase 2) - Native Node.js via @xenova/transformers
// ============================================================================

type Pipeline = (
  text: string,
  options?: { pooling?: string; normalize?: boolean },
) => Promise<{ data: Float32Array }>;

class NodeEmbedder {
  private pipeline: Pipeline | null = null;
  private loading: Promise<void> | null = null;
  private modelName: string;

  constructor(modelName: string) {
    this.modelName = modelName;
  }

  async initialize(): Promise<boolean> {
    if (this.pipeline) return true;
    if (this.loading) {
      await this.loading;
      return !!this.pipeline;
    }

    this.loading = (async () => {
      try {
        // Dynamic import to avoid loading if not needed
        const { pipeline } = await import("@xenova/transformers");
        this.pipeline = (await pipeline("feature-extraction", this.modelName)) as Pipeline;
      } catch (err) {
        console.error("Failed to load embedding model:", err);
        this.pipeline = null;
      }
    })();

    await this.loading;
    return !!this.pipeline;
  }

  async embed(text: string): Promise<number[] | null> {
    if (!this.pipeline) {
      const ok = await this.initialize();
      if (!ok) return null;
    }

    try {
      const output = await this.pipeline!(text, { pooling: "mean", normalize: true });
      return Array.from(output.data);
    } catch (err) {
      console.error("Embedding failed:", err);
      return null;
    }
  }

  isReady(): boolean {
    return !!this.pipeline;
  }
}

// ============================================================================
// Vector Math
// ============================================================================

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;

  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

// ============================================================================
// Binary Embedding Utilities (V3)
// ============================================================================

function encodeEmbedding(floats: number[]): string {
  const buf = Buffer.from(new Float32Array(floats).buffer);
  return buf.toString("base64");
}

function decodeEmbedding(b64: string): Float32Array {
  const buf = Buffer.from(b64, "base64");
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function getEmbeddingFloat32(entry: V2MemoryEntry): Float32Array | null {
  if (!entry.embedding) return null;
  if (typeof entry.embedding === "string") return decodeEmbedding(entry.embedding);
  return new Float32Array(entry.embedding);
}

function cosineSimilarityF32(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 0;
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

// ============================================================================
// Decay & Effective Importance
// ============================================================================

function effectiveImportance(entry: V2MemoryEntry): number {
  const daysSinceCreation =
    (Date.now() - new Date(entry.timestamp).getTime()) / (1000 * 60 * 60 * 24);
  const decayFactor = Math.max(0.3, 1 - daysSinceCreation / 365);
  const accessBoost = Math.min(2.0, 1 + (entry.accessCount || 0) * 0.1);
  return entry.importance * decayFactor * accessBoost;
}

// ============================================================================
// Graph Operations
// ============================================================================

function addRelation(
  entry: V2MemoryEntry,
  targetId: string,
  relType: MemoryRelationType,
): V2MemoryEntry {
  const relations = entry.relations || [];
  if (relations.some((r) => r.id === targetId && r.type === relType)) return entry;
  return { ...entry, relations: [...relations, { id: targetId, type: relType }] };
}

const INVERSE_RELATION: Record<MemoryRelationType, MemoryRelationType> = {
  caused: "caused_by",
  caused_by: "caused",
  related: "related",
  supersedes: "supersedes",
  contradicts: "contradicts",
  elaborates: "elaborates",
};

function searchWithGraph(
  allMemories: V2MemoryEntry[],
  results: Array<{ entry: V2MemoryEntry; score: number }>,
  depth: number = 1,
): Array<{ entry: V2MemoryEntry; score: number; related: V2MemoryEntry[] }> {
  const memoryById = new Map<string, V2MemoryEntry>();
  for (const m of allMemories) memoryById.set(m.id, m);

  const resultIds = new Set(results.map((r) => r.entry.id));

  return results.map(({ entry, score }) => {
    const related: V2MemoryEntry[] = [];
    const visited = new Set<string>([entry.id, ...resultIds]);

    let frontier = entry.relations || [];
    for (let d = 0; d < depth && frontier.length > 0; d++) {
      const nextFrontier: MemoryRelation[] = [];
      for (const rel of frontier) {
        if (visited.has(rel.id)) continue;
        visited.add(rel.id);
        const target = memoryById.get(rel.id);
        if (target) {
          related.push(target);
          if (target.relations) nextFrontier.push(...target.relations);
        }
      }
      frontier = nextFrontier;
    }

    // Increment access tracking
    entry.accessCount = (entry.accessCount || 0) + 1;
    entry.lastAccessed = new Date().toISOString();

    return { entry, score, related };
  });
}

function autoLinkNewEntry(
  newEntry: V2MemoryEntry,
  allMemories: V2MemoryEntry[],
  threshold: number = 0.65,
): V2MemoryEntry[] {
  const newEmb = getEmbeddingFloat32(newEntry);
  if (!newEmb) return allMemories;

  const scored: Array<{ idx: number; sim: number }> = [];
  for (let i = 0; i < allMemories.length; i++) {
    const m = allMemories[i];
    if (m.id === newEntry.id) continue;
    const emb = getEmbeddingFloat32(m);
    if (!emb) continue;
    const sim = cosineSimilarityF32(newEmb, emb);
    if (sim >= threshold) scored.push({ idx: i, sim });
  }

  scored.sort((a, b) => b.sim - a.sim);
  const topN = scored.slice(0, 3);

  const updated = [...allMemories];
  for (const { idx } of topN) {
    newEntry = addRelation(newEntry, updated[idx].id, "related");
    updated[idx] = addRelation(updated[idx], newEntry.id, "related");
  }

  return updated;
}

// ============================================================================
// Index Operations
// ============================================================================

function normalizeDate(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return new Date().toISOString().slice(0, 10);
  }
  return parsed.toISOString().slice(0, 10);
}

function ensureDate(entry: V2MemoryEntry): V2MemoryEntry {
  if (!entry.date) {
    return { ...entry, date: normalizeDate(entry.timestamp) };
  }
  return entry;
}

function loadIndex(indexPath: string): V2MemoryIndex {
  if (!fs.existsSync(indexPath)) {
    return { version: "2.1", lastUpdated: new Date().toISOString(), memories: [] };
  }

  try {
    const content = fs.readFileSync(indexPath, "utf-8");
    const lines = content.split(/\r?\n/).filter((line) => line.trim().length > 0);

    if (lines.length === 0) {
      return { version: "2.1", lastUpdated: new Date().toISOString(), memories: [] };
    }

    let version = "2.1";
    let lastUpdated = new Date().toISOString();
    let embeddingModel: string | undefined;
    let foundMeta = false;
    const memories: V2MemoryEntry[] = [];

    for (const line of lines) {
      try {
        const parsed = JSON.parse(line) as Record<string, unknown>;
        if (parsed && parsed._meta === true) {
          foundMeta = true;
          if (typeof parsed.version === "string") version = parsed.version;
          if (typeof parsed.lastUpdated === "string") lastUpdated = parsed.lastUpdated;
          if (typeof parsed.embeddingModel === "string") embeddingModel = parsed.embeddingModel;
          continue;
        }

        const entry = parsed as V2MemoryEntry;
        memories.push(ensureDate(entry));
      } catch (err) {
        console.warn("memory-v2: skipped malformed JSONL line", err);
      }
    }

    if (!foundMeta && memories.length === 0 && content.includes('"memories"')) {
      try {
        const legacy = JSON.parse(content) as V2MemoryIndex;
        return {
          version: legacy.version || "2.0",
          lastUpdated: legacy.lastUpdated || new Date().toISOString(),
          embeddingModel: legacy.embeddingModel,
          memories: (legacy.memories || []).map((entry) => ensureDate(entry)),
        };
      } catch (err) {
        console.warn("memory-v2: failed to parse legacy JSON index", err);
      }
    }

    return { version, lastUpdated, memories, embeddingModel };
  } catch (err) {
    console.warn("memory-v2: failed to read index", err);
    return { version: "2.1", lastUpdated: new Date().toISOString(), memories: [] };
  }
}

function saveIndex(indexPath: string, index: V2MemoryIndex): void {
  const dir = path.dirname(indexPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  if (!index.version) {
    index.version = "2.1";
  }
  index.lastUpdated = new Date().toISOString();

  const metaLine = JSON.stringify({
    _meta: true,
    version: "3.0",
    lastUpdated: index.lastUpdated,
    embeddingModel: index.embeddingModel,
    embeddingFormat: "base64-f32",
    dimensions: 384,
  });

  const lines = [metaLine];
  for (const memory of index.memories) {
    const entry = ensureDate(memory);
    const serialized =
      entry.embedding && Array.isArray(entry.embedding)
        ? { ...entry, embedding: encodeEmbedding(entry.embedding) }
        : entry;
    lines.push(JSON.stringify(serialized));
  }

  fs.writeFileSync(indexPath, `${lines.join("\n")}\n`);
}

function appendEntry(indexPath: string, entry: V2MemoryEntry): void {
  const dir = path.dirname(indexPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  const normalized = ensureDate(entry);
  const serialized =
    normalized.embedding && Array.isArray(normalized.embedding)
      ? { ...normalized, embedding: encodeEmbedding(normalized.embedding) }
      : normalized;

  if (!fs.existsSync(indexPath) || fs.statSync(indexPath).size === 0) {
    const metaLine = JSON.stringify({
      _meta: true,
      version: "3.0",
      lastUpdated: new Date().toISOString(),
      embeddingFormat: "base64-f32",
      dimensions: 384,
    });
    fs.writeFileSync(indexPath, `${metaLine}\n${JSON.stringify(serialized)}\n`);
    return;
  }

  fs.appendFileSync(indexPath, `\n${JSON.stringify(serialized)}`);
}

function migrateJsonToJsonl(jsonPath: string, jsonlPath: string): boolean {
  if (!fs.existsSync(jsonPath) || fs.existsSync(jsonlPath)) {
    return false;
  }

  try {
    const raw = fs.readFileSync(jsonPath, "utf-8");
    const legacy = JSON.parse(raw) as V2MemoryIndex;
    const memories = (legacy.memories || []).map((entry) => ensureDate(entry as V2MemoryEntry));

    const migrated: V2MemoryIndex = {
      version: "2.1",
      lastUpdated: legacy.lastUpdated || new Date().toISOString(),
      embeddingModel: legacy.embeddingModel,
      memories,
    };

    saveIndex(jsonlPath, migrated);
    return true;
  } catch (err) {
    console.warn("memory-v2: migration failed", err);
    return false;
  }
}

// ============================================================================
// Search
// ============================================================================

function scoreKeywordMatch(entry: V2MemoryEntry, queryTerms: string[]): number {
  const content = entry.content.toLowerCase();
  const tags = entry.tags.map((t) => t.toLowerCase());
  const type = entry.type.toLowerCase();

  let score = 0;
  let matchedTerms = 0;

  for (const term of queryTerms) {
    const termLower = term.toLowerCase();
    if (content.includes(termLower)) {
      score += 0.4;
      matchedTerms++;
    }
    if (tags.some((tag) => tag.includes(termLower))) {
      score += 0.3;
      matchedTerms++;
    }
    if (type.includes(termLower)) {
      score += 0.2;
      matchedTerms++;
    }
  }

  if (queryTerms.length > 0) {
    score = score / queryTerms.length;
  }

  score += (entry.importance / 10) * 0.2;

  if (matchedTerms >= queryTerms.length) {
    score += 0.1;
  }

  return Math.min(score, 1.0);
}

function searchMemories(
  indexPath: string,
  query: string,
  maxResults: number,
  minScore: number,
  queryEmbedding?: number[] | null,
): Array<{ entry: V2MemoryEntry; score: number; keywordScore: number; semanticScore: number }> {
  const index = loadIndex(indexPath);
  const queryTerms = query
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 1);

  if (queryTerms.length === 0 && !queryEmbedding) {
    return [];
  }

  const scored: Array<{
    entry: V2MemoryEntry;
    score: number;
    keywordScore: number;
    semanticScore: number;
  }> = [];

  // Convert query embedding to Float32Array once for efficient comparison
  const queryEmbF32 = queryEmbedding ? new Float32Array(queryEmbedding) : null;

  for (const entry of index.memories) {
    const keywordScore = queryTerms.length > 0 ? scoreKeywordMatch(entry, queryTerms) : 0;

    let semanticScore = 0;
    if (queryEmbF32 && entry.embedding) {
      const entryEmb = getEmbeddingFloat32(entry);
      if (entryEmb) {
        semanticScore = cosineSimilarityF32(queryEmbF32, entryEmb);
      }
    }

    let score: number;
    if (queryEmbF32 && entry.embedding) {
      // Hybrid: 60% semantic, 40% keyword
      score = semanticScore * 0.6 + keywordScore * 0.4;
    } else {
      score = keywordScore;
    }

    // Use decay-aware importance instead of raw importance
    score += (effectiveImportance(entry) / 10) * 0.1;

    if (score >= minScore) {
      scored.push({ entry, score, keywordScore, semanticScore });
    }
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, maxResults);
}

// ============================================================================
// Plugin
// ============================================================================

const memoryV2Plugin = {
  id: "memory-v2",
  name: "Memory V2",
  description: "Structured memory with typed entries, importance scores, tags, and semantic search",
  kind: "memory" as const,

  configSchema: {
    type: "object" as const,
    additionalProperties: false,
    properties: {
      indexPath: { type: "string" },
      embedding: {
        type: "object",
        additionalProperties: false,
        properties: {
          provider: { type: "string", enum: ["local", "none"] },
          modelName: { type: "string" },
        },
      },
    },
  },

  register(api: OpenClawPluginApi) {
    const config = parseConfig(api.pluginConfig);
    let indexPath = config.indexPath;
    let jsonPath: string | null = null;
    let jsonlPath: string | null = null;

    if (indexPath.endsWith(".jsonl")) {
      jsonlPath = indexPath;
      jsonPath = indexPath.replace(/\.jsonl$/, ".json");
    } else if (indexPath.endsWith(".json")) {
      jsonPath = indexPath;
      jsonlPath = `${indexPath}l`;
      indexPath = jsonlPath;
    }

    if (jsonPath && jsonlPath) {
      const migrated = migrateJsonToJsonl(jsonPath, jsonlPath);
      if (migrated) {
        api.logger.info(`memory-v2: migrated index to JSONL at ${jsonlPath}`);
      }
    }

    // Initialize embedder if configured
    let embedder: NodeEmbedder | null = null;
    if (config.embedding.provider === "local") {
      embedder = new NodeEmbedder(config.embedding.modelName || "Xenova/all-MiniLM-L6-v2");
      // Pre-warm the model in background
      embedder.initialize().then((ok) => {
        if (ok) api.logger.info("memory-v2: embedding model loaded");
        else
          api.logger.warn(
            "memory-v2: embedding model failed to load, falling back to keyword search",
          );
      });
    }

    api.logger.info(`memory-v2: plugin loaded (embeddings: ${config.embedding.provider})`);

    // ========================================================================
    // Tool: memory_v2_search
    // ========================================================================
    api.registerTool(
      {
        name: "memory_v2_search",
        label: "Memory Search (V2)",
        description:
          "Search structured memories. Uses hybrid keyword + semantic search when embeddings are enabled.",
        parameters: Type.Object({
          query: Type.String({ description: "Search query" }),
          maxResults: Type.Optional(Type.Number({ description: "Max results (default: 10)" })),
          minScore: Type.Optional(
            Type.Number({ description: "Min score threshold (default: 0.1)" }),
          ),
        }),
        async execute(_toolCallId, params) {
          const query = typeof params.query === "string" ? params.query : "";
          const maxResults = typeof params.maxResults === "number" ? params.maxResults : 10;
          const minScore = typeof params.minScore === "number" ? params.minScore : 0.1;

          // Get query embedding if available
          let queryEmbedding: number[] | null = null;
          if (embedder) {
            queryEmbedding = await embedder.embed(query);
          }

          const results = searchMemories(indexPath, query, maxResults, minScore, queryEmbedding);

          // Graph-aware: enrich results with related memories
          const index = loadIndex(indexPath);
          const graphResults = searchWithGraph(index.memories, results, 1);
          // Persist access tracking
          saveIndex(indexPath, index);

          const searchMode = queryEmbedding ? "hybrid" : "keyword";
          const formatted = graphResults.map(({ entry, score, related }) => ({
            id: entry.id,
            path: entry.file.startsWith("daily/") ? `memory/${entry.file}` : entry.file,
            line: entry.line,
            score: Math.round(score * 100) / 100,
            type: entry.type,
            importance: entry.importance,
            effectiveImportance: Math.round(effectiveImportance(entry) * 100) / 100,
            tags: entry.tags,
            snippet: entry.content,
            hasEmbedding: !!entry.embedding,
            relations: entry.relations || [],
            relatedMemories: related.map((r) => ({
              id: r.id,
              type: r.type,
              snippet: r.content.slice(0, 80),
            })),
          }));

          return {
            content: [
              {
                type: "text",
                text:
                  results.length > 0
                    ? `Found ${results.length} memories (${searchMode}):\n\n${graphResults
                        .map((r, i) => {
                          let line = `${i + 1}. [${r.entry.type.toUpperCase()}] (importance: ${r.entry.importance}, score: ${(r.score * 100).toFixed(0)}%) ${r.entry.content.slice(0, 100)}...`;
                          if (r.related.length > 0) {
                            line += `\n   └─ ${r.related.length} related: ${r.related.map((rel) => rel.content.slice(0, 50)).join("; ")}`;
                          }
                          return line;
                        })
                        .join("\n")}`
                    : `No matching memories found (${searchMode}).`,
              },
            ],
            details: { results: formatted, searchMode },
          };
        },
      },
      { name: "memory_v2_search" },
    );

    // ========================================================================
    // Tool: memory_v2_get
    // ========================================================================
    api.registerTool(
      {
        name: "memory_v2_get",
        label: "Memory Get (V2)",
        description: "Read content from a memory file.",
        parameters: Type.Object({
          path: Type.String({ description: "Path to memory file" }),
          from: Type.Optional(Type.Number({ description: "Starting line number" })),
          lines: Type.Optional(Type.Number({ description: "Number of lines to read" })),
        }),
        async execute(_toolCallId, params) {
          const relPath = typeof params.path === "string" ? params.path : "";
          const from = typeof params.from === "number" ? params.from : undefined;
          const numLines = typeof params.lines === "number" ? params.lines : undefined;

          let absPath = relPath;
          if (relPath.startsWith("~")) {
            absPath = path.join(homedir(), relPath.slice(1));
          } else if (!path.isAbsolute(relPath)) {
            absPath = path.join(homedir(), ".openclaw", "workspace", relPath);
          }

          try {
            const content = fs.readFileSync(absPath, "utf-8");
            const allLines = content.split("\n");

            let text: string;
            if (from !== undefined || numLines !== undefined) {
              const startLine = (from ?? 1) - 1;
              const endLine = numLines ? startLine + numLines : allLines.length;
              text = allLines.slice(startLine, endLine).join("\n");
            } else {
              text = content;
            }

            return {
              content: [{ type: "text", text }],
              details: { path: relPath },
            };
          } catch {
            return {
              content: [{ type: "text", text: `Could not read file: ${relPath}` }],
              details: { path: relPath, error: true },
            };
          }
        },
      },
      { name: "memory_v2_get" },
    );

    // ========================================================================
    // Tool: memory_v2_embed
    // ========================================================================
    api.registerTool(
      {
        name: "memory_v2_embed",
        label: "Memory Embed (V2)",
        description: "Generate embeddings for memories. Run this to enable semantic search.",
        parameters: Type.Object({
          limit: Type.Optional(
            Type.Number({ description: "Max memories to embed (default: 100)" }),
          ),
          force: Type.Optional(Type.Boolean({ description: "Re-embed even if embedding exists" })),
        }),
        async execute(_toolCallId, params) {
          if (!embedder) {
            return {
              content: [
                {
                  type: "text",
                  text: "Embeddings not configured. Set embedding.provider to 'local' in plugin config.",
                },
              ],
              details: { error: true },
            };
          }

          const ready = await embedder.initialize();
          if (!ready) {
            return {
              content: [
                {
                  type: "text",
                  text: "Failed to load embedding model. Check that @xenova/transformers is installed.",
                },
              ],
              details: { error: true },
            };
          }

          const limit = typeof params.limit === "number" ? params.limit : 100;
          const force = params.force === true;

          const index = loadIndex(indexPath);
          let embedded = 0,
            skipped = 0,
            failed = 0;

          for (const memory of index.memories) {
            if (embedded >= limit) break;

            if (memory.embedding && !force) {
              skipped++;
              continue;
            }

            const embedding = await embedder.embed(memory.content);
            if (embedding) {
              memory.embedding = embedding;
              embedded++;
            } else {
              failed++;
            }
          }

          // Auto-link newly embedded entries to similar existing ones
          let linked = 0;
          if (embedded > 0) {
            for (const memory of index.memories) {
              if (!memory.relations || memory.relations.length === 0) {
                const before = memory.relations?.length || 0;
                const updated = autoLinkNewEntry(memory, index.memories);
                // Copy updated relations back
                for (let i = 0; i < updated.length; i++) {
                  index.memories[i] = updated[i];
                }
                const after = memory.relations?.length || 0;
                if (after > before) linked++;
              }
            }

            index.embeddingModel = config.embedding.modelName;
            saveIndex(indexPath, index);
          }

          return {
            content: [
              {
                type: "text",
                text: `Embedding complete: ${embedded} embedded, ${skipped} already had embeddings, ${failed} failed.${linked > 0 ? ` Auto-linked ${linked} memories.` : ""}\nSemantic search is now ${embedded > 0 || skipped > 0 ? "enabled" : "disabled"}.`,
              },
            ],
            details: { embedded, skipped, failed, linked, model: config.embedding.modelName },
          };
        },
      },
      { name: "memory_v2_embed" },
    );

    // ========================================================================
    // Tool: memory_v2_stats
    // ========================================================================
    api.registerTool(
      {
        name: "memory_v2_stats",
        label: "Memory Stats (V2)",
        description: "Show memory index statistics.",
        parameters: Type.Object({}),
        async execute() {
          const index = loadIndex(indexPath);

          const byType: Record<string, number> = {};
          const byImportance: Record<string, number> = {};
          let withEmbeddings = 0;

          for (const m of index.memories) {
            byType[m.type] = (byType[m.type] || 0) + 1;
            const bucket =
              m.importance >= 8 ? "high (8-10)" : m.importance >= 5 ? "medium (5-7)" : "low (1-4)";
            byImportance[bucket] = (byImportance[bucket] || 0) + 1;
            if (m.embedding) withEmbeddings++;
          }

          const pct =
            index.memories.length > 0
              ? Math.round((withEmbeddings / index.memories.length) * 100)
              : 0;

          return {
            content: [
              {
                type: "text",
                text: `Memory V2 Stats:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total memories: ${index.memories.length}
With embeddings: ${withEmbeddings}/${index.memories.length} (${pct}%)
Embedding model: ${index.embeddingModel || "none"}
Provider: ${config.embedding.provider}

By type: ${Object.entries(byType)
                  .map(([k, v]) => `${k}:${v}`)
                  .join(", ")}
By importance: ${Object.entries(byImportance)
                  .map(([k, v]) => `${k}:${v}`)
                  .join(", ")}

${pct < 100 && config.embedding.provider === "local" ? "💡 Run memory_v2_embed to enable semantic search for all memories." : ""}`,
              },
            ],
            details: {
              total: index.memories.length,
              withEmbeddings,
              embeddingModel: index.embeddingModel,
              provider: config.embedding.provider,
              byType,
              byImportance,
            },
          };
        },
      },
      { name: "memory_v2_stats" },
    );

    // ========================================================================
    // Tool: memory_v2_migrate (v2 → v3 binary embeddings)
    // ========================================================================
    api.registerTool(
      {
        name: "memory_v2_migrate",
        label: "Migrate to V3 (Binary Embeddings)",
        description:
          "Migrate memory index from v2 (JSON float arrays) to v3 (base64 binary embeddings). Reduces file size ~42%.",
        parameters: Type.Object({}),
        async execute() {
          const index = loadIndex(indexPath);
          let migrated = 0;
          let alreadyBinary = 0;
          const sizeBefore = fs.existsSync(indexPath) ? fs.statSync(indexPath).size : 0;

          for (const memory of index.memories) {
            if (memory.embedding && Array.isArray(memory.embedding)) {
              memory.embedding = encodeEmbedding(memory.embedding);
              migrated++;
            } else if (memory.embedding && typeof memory.embedding === "string") {
              alreadyBinary++;
            }
          }

          index.version = "3.0";
          saveIndex(indexPath, index);
          const sizeAfter = fs.statSync(indexPath).size;
          const reduction = sizeBefore > 0 ? Math.round((1 - sizeAfter / sizeBefore) * 100) : 0;

          return {
            content: [
              {
                type: "text",
                text: `Migration complete:\n- Converted: ${migrated} embeddings to base64\n- Already binary: ${alreadyBinary}\n- Size before: ${(sizeBefore / 1024).toFixed(1)}KB\n- Size after: ${(sizeAfter / 1024).toFixed(1)}KB\n- Reduction: ${reduction}%`,
              },
            ],
            details: { migrated, alreadyBinary, sizeBefore, sizeAfter, reduction },
          };
        },
      },
      { name: "memory_v2_migrate" },
    );

    // ========================================================================
    // Tool: memory_v2_link (create relations between memories)
    // ========================================================================
    api.registerTool(
      {
        name: "memory_v2_link",
        label: "Link Memories",
        description:
          "Create a bidirectional relation between two memories. Types: caused, caused_by, related, supersedes, contradicts, elaborates.",
        parameters: Type.Object({
          sourceId: Type.String({ description: "Source memory ID" }),
          targetId: Type.String({ description: "Target memory ID" }),
          relationType: Type.String({
            description: "Relation type",
            enum: ["caused", "caused_by", "related", "supersedes", "contradicts", "elaborates"],
          }),
        }),
        async execute(
          _toolCallId: string,
          params: { sourceId: string; targetId: string; relationType: string },
        ) {
          const index = loadIndex(indexPath);
          const relType = params.relationType as MemoryRelationType;
          const inverseType = INVERSE_RELATION[relType];

          let sourceFound = false;
          let targetFound = false;

          for (let i = 0; i < index.memories.length; i++) {
            if (index.memories[i].id === params.sourceId) {
              index.memories[i] = addRelation(index.memories[i], params.targetId, relType);
              sourceFound = true;
            }
            if (index.memories[i].id === params.targetId) {
              index.memories[i] = addRelation(index.memories[i], params.sourceId, inverseType);
              targetFound = true;
            }
          }

          if (!sourceFound || !targetFound) {
            return {
              content: [
                {
                  type: "text",
                  text: `Error: ${!sourceFound ? `Source '${params.sourceId}' not found.` : ""} ${!targetFound ? `Target '${params.targetId}' not found.` : ""}`.trim(),
                },
              ],
              details: { error: true },
            };
          }

          saveIndex(indexPath, index);
          return {
            content: [
              {
                type: "text",
                text: `Linked: ${params.sourceId} —[${relType}]→ ${params.targetId} (bidirectional: ${inverseType})`,
              },
            ],
            details: { sourceId: params.sourceId, targetId: params.targetId, relType, inverseType },
          };
        },
      },
      { name: "memory_v2_link" },
    );
  },
};

export default memoryV2Plugin;
