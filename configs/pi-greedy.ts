import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.on("before_provider_request", (event, ctx) => {
    if (ctx.model?.provider !== "inkling-local") return;
    return { ...event.payload, temperature: 0, seed: 17, max_tokens: 512 };
  });
}
