import fs from "node:fs";
export default function (pi) {
  const gate = JSON.parse(fs.readFileSync(process.env.FIXTURE_GATE, "utf8"));
  pi.on("before_provider_request", (event) => {
    const p = event.payload;
    for (const m of p.messages) {
      if (m.role === "system" && typeof m.content === "string")
        m.content = m.content.replaceAll(process.cwd(), "/fixture");
    }
    return { ...p, cache_prompt: p.messages.some(m => m.role === "tool") || p.messages.filter(m => m.role === "user").length > 1, temperature: 0, max_tokens: 512, seed: 17, stream_options: { include_usage: true } };
  });
  let attempts = 0;
  pi.on("tool_call", (e) => {
    if (++attempts > 12) return { block: true, terminate: true, reason: "Fixture exceeded twelve tool attempts." };
    const a = e.input;
    const path = a.path?.replace(/^\.\//, "");
    let ok = false;
    if (e.toolName === "read") ok = path === "calc.py" || path === "check.py";
    if (e.toolName === "edit") ok = path === "calc.py" && a.edits?.length === 1
      && typeof a.edits[0].oldText === "string" && a.edits[0].oldText.length > 0
      && gate.before.includes(a.edits[0].oldText)
      && gate.before.replace(a.edits[0].oldText, () => a.edits[0].newText) === gate.after;
    if (e.toolName === "bash") ok = a.command === "python3 -B check.py"
      || /^(?:pwd|ls(?: -[alh]+)?(?: (?:\.|calc\.py|check\.py))?|(?:cat(?: -n)?|nl(?: -ba)?) (?:calc|check)\.py)$/.test(a.command)
      || a.command === 'grep -n "' + gate.old + '" calc.py';
    if (!ok) return {block: true, reason: "This fixture permits inspection, the requested change through edit, and the specified test through bash."};
  });
}
