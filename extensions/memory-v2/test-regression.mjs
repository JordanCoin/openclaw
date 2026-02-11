import { createJiti } from "jiti";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const jiti = createJiti(fileURLToPath(import.meta.url));
const pluginMod = jiti("./index.ts");
const plugin = pluginMod.default;

function buildJsonl(entries) {
  const meta = {
    _meta: true,
    version: "3.0",
    lastUpdated: new Date().toISOString(),
    embeddingModel: "Xenova/all-MiniLM-L6-v2",
    embeddingFormat: "base64-f32",
    dimensions: 384,
  };
  return `${JSON.stringify(meta)}\n${entries.map((e) => JSON.stringify(e)).join("\n")}\n`;
}

function registerWithIndex(indexPath) {
  const tools = new Map();
  const api = {
    pluginConfig: {
      indexPath,
      embedding: { provider: "none" },
    },
    logger: {
      info: () => {},
      warn: () => {},
      error: () => {},
    },
    registerTool: (def) => {
      tools.set(def.name, def);
    },
  };

  plugin.register(api);
  return tools;
}

async function run() {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "memory-v2-regression-"));

  // ---------------------------------------------------------------------------
  // Test 1: memory_v2_search should NOT crash when entries are missing `file`.
  // ---------------------------------------------------------------------------
  const idx1 = path.join(tmpDir, "missing-file.jsonl");
  fs.writeFileSync(
    idx1,
    buildJsonl([
      {
        id: "m1",
        timestamp: "2026-02-11T12:00:00Z",
        type: "learning",
        importance: 7,
        content: "calendar api enabled and tested",
        tags: ["calendar", "api"],
        // file intentionally missing
        line: 10,
      },
      {
        id: "m2",
        timestamp: "2026-02-11T13:00:00Z",
        type: "event",
        importance: 6,
        content: "entry without tags and line",
        // tags/line intentionally missing
      },
    ]),
  );

  const tools1 = registerWithIndex(idx1);
  const search1 = tools1.get("memory_v2_search");
  assert.ok(search1, "memory_v2_search tool should register");

  const res1 = await search1.execute("tc-1", {
    query: "calendar api",
    maxResults: 5,
  });

  assert.ok(res1?.details?.results, "search should return details.results");
  assert.ok(res1.details.results.length > 0, "search should return at least one result");
  assert.equal(
    res1.details.results[0].path,
    "",
    "missing file should map to empty path instead of crashing",
  );

  // ---------------------------------------------------------------------------
  // Test 2: config path ending in .json should still read sibling .jsonl index.
  // ---------------------------------------------------------------------------
  const jsonPath = path.join(tmpDir, "memory-index.json");
  const jsonlPath = `${jsonPath}l`;

  fs.writeFileSync(jsonPath, ""); // stale empty legacy file
  fs.writeFileSync(
    jsonlPath,
    buildJsonl([
      {
        id: "m3",
        timestamp: "2026-02-11T14:00:00Z",
        type: "decision",
        importance: 8,
        content: "swtpa focus block moved to work calendar",
        tags: ["calendar", "swtpa"],
        file: "daily/2026-02-11.md",
        line: 42,
      },
    ]),
  );

  const tools2 = registerWithIndex(jsonPath); // intentionally .json, plugin should redirect to .jsonl
  const search2 = tools2.get("memory_v2_search");
  assert.ok(search2, "memory_v2_search tool should register for .json config path");

  const res2 = await search2.execute("tc-2", {
    query: "swtpa focus",
    maxResults: 5,
  });

  assert.ok(res2?.details?.results?.length > 0, "search should read entries from sibling .jsonl");
  assert.equal(
    res2.details.results[0].path,
    "memory/daily/2026-02-11.md",
    "daily/* paths should be expanded to memory/daily/*",
  );

  console.log("✅ memory-v2 regression tests passed");
}

run().catch((err) => {
  console.error("❌ memory-v2 regression tests failed");
  console.error(err);
  process.exit(1);
});
