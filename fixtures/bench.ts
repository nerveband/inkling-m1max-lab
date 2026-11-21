import fs from "node:fs";
export default function (pi) {
  const gate = JSON.parse(fs.readFileSync(process.env.FIXTURE_GATE, "utf8"));
  pi.on("before_provider_request", (event) => {
    const p = event.payload;
    for (const m of p.messages) {
      if (m.role === "system" && typeof m.content === "string")
        m.content = m.content.replaceAll(process.cwd(), "/fixture");
    }
    return { ...p, cache_prompt: p.messages.some(m => m.role === "tool"), temperature: 0, max_tokens: 512, seed: 17, stream_options: { include_usage: true } };
  });
  pi.on("tool_call", (e) => {
    const a = e.input;
    const path = a.path?.replace(/^\.\//, "");
    let ok = false;
    if (e.toolName === "read") ok = path === "calc.py";
    if (e.toolName === "edit") ok = path === "calc.py" && a.edits?.length === 1
      && typeof a.edits[0].oldText === "string" && a.edits[0].oldText.length > 0
      && gate.before.includes(a.edits[0].oldText)
      && gate.before.replace(a.edits[0].oldText, () => a.edits[0].newText) === gate.after;
    if (e.toolName === "bash") ok = a.command === "python3 -B check.py"
      || a.command === 'grep -n "' + gate.old + '" calc.py';
    if (!ok) return {block: true, terminate: true, reason: "Fixture tool arguments failed."};
  });
}
