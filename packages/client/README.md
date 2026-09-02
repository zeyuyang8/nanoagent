# `@nanoagent/client`

Typed, dependency-free Node client for a server-hosted NanoAgent runtime.
Its request and event types are generated from the same JSON Schemas the Python host validates,
so the two packages share one wire contract.

```ts
import {NanoAgentClient} from "@nanoagent/client";

const nanoagent = new NanoAgentClient({
  baseUrl: process.env.NANOAGENT_URL!,
  token: process.env.NANOAGENT_API_TOKEN,
});

for await (const event of nanoagent.stream(
  {input: "Summarize this", messages: priorMessages, instructions: workspaceAgent.instructions},
  {signal: requestAbortSignal},
)) {
  if (event.type === "delta" && event.kind === "content") {
    sendToBrowser(event.text);
  }
}
```

Use this package from the application server. Do not put the NanoAgent URL or bearer token in a
browser bundle; the application server remains the authentication, tenancy and policy boundary.
