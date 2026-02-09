import { createJiti } from "jiti";
import { fileURLToPath } from "node:url";

const jiti = createJiti(fileURLToPath(import.meta.url));

try {
  const plugin = jiti("./index.ts");
  console.log("✅ Plugin loaded successfully");
  console.log("ID:", plugin.default?.id);
  console.log("Kind:", plugin.default?.kind);
  console.log("Has register:", typeof plugin.default?.register === "function");
} catch (e) {
  console.log("❌ Plugin failed to load:");
  console.log(e.message);
  console.log(e.stack);
}
