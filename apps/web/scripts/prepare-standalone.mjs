import { cp, mkdir, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const standalone = path.join(webRoot, ".next", "standalone");
const staticSource = path.join(webRoot, ".next", "static");
const staticTarget = path.join(standalone, ".next", "static");
const publicSource = path.join(webRoot, "public");
const publicTarget = path.join(standalone, "public");

if (!existsSync(standalone)) {
  throw new Error("Next.js standalone output was not generated.");
}

await rm(staticTarget, { recursive: true, force: true });
await mkdir(path.dirname(staticTarget), { recursive: true });
await cp(staticSource, staticTarget, { recursive: true });

if (existsSync(publicSource)) {
  await rm(publicTarget, { recursive: true, force: true });
  await cp(publicSource, publicTarget, { recursive: true });
}
