import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const base = "http://127.0.0.1:18089";

async function local(path: string, body: unknown) {
  const res = await fetch(base + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Prefix preparation failed at ${path} (HTTP ${res.status})`);
  return res.json();
}

export default function (pi: ExtensionAPI) {
  let warmed = false;
  pi.on("before_provider_request", async (event, ctx) => {
    if (warmed || ctx.model?.provider !== "inkling-local") return;
    const request = event.payload as {
      model: string;
      tools?: unknown[];
      messages: { role: string; content: unknown }[];
    };
    const system = request.messages.filter((m) => m.role === "system");
    const shape = (word: string) => ({
      model: request.model,
      tools: request.tools,
      messages: [...system, { role: "user", content: word }],
    });
    const a = (await local("/tokenize", shape("Alpine."))).tokens as number[];
    const b = (await local("/tokenize", shape("Zebra."))).tokens as number[];
    let shared = 0;
    while (shared < a.length && shared < b.length && a[shared] === b[shared]) shared++;
    if (shared === 0 || shared === a.length) return;
    const ids = a.slice(0, shared + 1);
    const { prompt } = await local("/detokenize", { model: request.model, tokens: ids });
    const encoded = (await local("/tokenize", { model: request.model, prompt })).tokens as number[];
    if (encoded.length !== ids.length || ids.some((id, i) => id !== encoded[i])) return;
    // oMLX snapshots this stateful model at input length minus one.
    await local("/v1/completions", { model: request.model, prompt, max_tokens: 1, temperature: 0 });
    warmed = true;
  });
}
