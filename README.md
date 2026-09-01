# nanoagent

A minimal, clean-room agent loop with structured tool calling — plus the batch runner, REPL and
coding-CLI built on top of it.

The whole loop is one function. A model is anything with an `async query(messages, tools)`; a tool
is a class with a `NAME`, a JSON Schema and a `run`. Everything else in the package is a *seam*
onto that loop, off unless a config names it — so the path a batch of 512 concurrent rollouts
takes is the same code an interactive chat takes.

```python
import asyncio
from nanoagent import Agent, get_tools
from nanoagent.harness.core.model import Model
from nanoagent.harness.config import ModelConfig

tools = get_tools(["tools/bash.yaml"])                 # shipped with the package
agent = Agent(model=Model.from_config(cfg), tools=tools, system_prompt="...")
result = asyncio.run(agent.run("count the python files under src/"))
print(result.answer)
```

## Install

```bash
pip install git+https://github.com/zeyuyang8/nanoagent

# ...plus `[serve]` if you also want to bring the SGLang server up yourself
pip install "nanoagent[serve] @ git+https://github.com/zeyuyang8/nanoagent"
```

## The four entry points

Everything is described by a YAML file rather than by flags, because a run has to be reproducible
from something you can commit. The one exception is `mgen`, and it is a thin layer *over* a config
(see below).

```bash
# one task, streamed live, trajectory always saved
nanoagent run harness_cfg=myharness.yaml task="list the python files" output=run.traj.json

# a batch over a tasks JSONL — concurrent, resumable, one bad task never sinks the rest
nanoagent run harness_cfg=myharness.yaml batch_cfg=mybatch.yaml \
    tasks=mytasks.jsonl output=expdir/batch

# an interactive session over the same loop (confirm / yolo / shell modes, branching transcript)
nanoagent chat chat_cfg=mychat.yaml

# a TUI for reading back what any of the above wrote
nanoagent browse path=expdir/chat/
```

Any leaf of a config can be overridden inline on any command:

```bash
nanoagent run harness_cfg=myharness.yaml agent.max_steps=30 model.temperature=0.7
```

### `mgen` — the same agent as a coding CLI

`mgen` speaks Claude Code's flag grammar, so a script written against `claude` runs against
whatever you can reach by changing a config rather than the script:

```bash
mgen -p "summarise what changed in the last commit"
mgen --model my-model --allowedTools bash,read
mgen -p "..." --output-format json | jq -r .result
mgen -c                      # carry on from the last session
mgen --dump-config           # what those flags resolved to, as a YAML that reproduces them
```

The flags only override leaves that already exist in the schema, and `--dump-config` prints the
result — so nothing a flag can express is unsayable in a config. With no `--config` and no
`$MGEN_CONFIG`, it loads the `configs/mgen.yaml` shipped inside the package, which points at a
local SGLang endpoint and turns on read / write / edit / bash / python.

## Where the model comes from

The agent loop imports no provider SDK. `nanoagent.harness.core.model.Model` is a thin adapter over
`nanoagent.inference`, which resolves `model.backend` against its own built-in transports and then
against the plugin directories in `$NANOAGENT_PLUGINS`:

```yaml
model:
  model: default
  backend: sglang                            # or the name of a plugin file
  base_url: http://127.0.0.1:30000/v1
```

So a private gateway is a `.py` file in a `$NANOAGENT_PLUGINS` directory — one that defines a
`BACKEND` class with a `from_config` classmethod. Nothing in the package needs a line changed, and
there is deliberately no allowlist of backend names.

## Serving

`nanoagent.inference` also brings the server up, so the same YAML describes both sides of the
connection. One `SGLangServeConfig` covers every topology and `mode` picks which:

```bash
pip install "nanoagent[serve] @ git+https://github.com/zeyuyang8/nanoagent"

python -m nanoagent.inference.serve --config configs/gemma_4_31b_serve.yaml    # single node
python -m nanoagent.inference.serve --config configs/gemma_4_31b_router.yaml   # router + N engines
```

SGLang itself is not a dependency: the engine is exec'd as the `sglang serve` CLI, so the serving
environment owns that pin (it decides the CUDA/torch stack) and this package never constrains it.
The four files in the repo-root `configs/` are worked examples — unlike `src/nanoagent/harness/configs/`,
which ships inside the wheel as `mgen`'s defaults.

For batch inference without an agent, `nanoagent.inference.infer` runs a list of requests
concurrently and returns the responses in input order.

## Tools

A tool is a `Tool` subclass; a tool *manifest* is a YAML naming the module that defines it. Every
other key in the manifest is passed to the constructor, so a tool's configuration is explicit in
the file rather than read out of the environment:

```yaml
# mytools/search.yaml
code: mytools/search.py     # a .py path (CWD-relative), or an importable module name
base_url: http://127.0.0.1:8000
```

Every concrete `Tool` subclass *defined* in that module becomes a tool. Adding one is a new file
and a new YAML — no change to nanoagent. The shipped manifests (`tools/files.yaml`,
`tools/bash.yaml`, `tools/python.yaml`) use the module-name form, since an installed package has
no repo-relative path to point at.

`tools.write` lets the agent write a manifest and its module and use the result on the next turn.
That works for the same reason: `get_tools` needs nothing but the two files.

## Configs are explicit

Every schema field is required. A config sets each one — `null` or `[]` to turn something off —
so nothing is inherited silently and no hidden default decides a run. Reuse comes from
`defaults: [<path>]` composition, not from omission.

## Layout

There are two things in this package: **reaching a model**, and **doing something with one**.

```text
src/nanoagent/
├── inference/     how a model is reached
└── harness/       what is done with one
```

| package | what it is |
| --- | --- |
| `nanoagent.inference` | the transport backends and their plugin resolver, the concurrent batch `engine`, and the SGLang `serve` / `router` / `launch` side. Needs none of the harness. |
| `nanoagent.harness.core` | the loop and what it is made of: `agent`, `tool`, `model`, plus the `hooks` / `events` / `workspace` seams. This is the rollout hot path. |
| `nanoagent.harness.tools` | the tools: `bash`, `code`, `file`, `write`, `skill`. None is loaded unless a manifest names it. |
| `nanoagent.harness.run` | a config becomes a run: `build`, `batch` (fan-out + resume ledger), `progress`, `trajectory`, `cli`, `mgen`. |
| `nanoagent.harness.repl` | chat only: `app`, `commands`, `tree` (a branching transcript), `browser`. |

The dependency arrows only ever point left, and the one arrow that crosses between the two halves
goes through a single module: `harness.core.model`. Nothing else in the harness imports
`inference`, and `inference` imports nothing from the harness at all.

## Seams

Each is off unless a config names it, so the rollout path executes the same code it always did.

- **hooks** — `session_start` / `before_llm` / `before_tool` / `after_tool` from a plain `.py`.
  This is how a prompt rule ("call search at most once") becomes something enforced rather than
  requested. State is per-*run*, not per-agent, so a shared `Agent` across concurrent rollouts
  gets one budget each.
- **skills** — `SKILL.md` files indexed by name and description, body loaded on demand.
- **events** — the run mirrored to NDJSON, for watching a rollout from outside the process.
- **workspace** — a per-rollout root that `Read` / `Write` / `Edit` resolve inside.

## Trajectories

Every run saves one. A trajectory is the full message list plus per-step timings, token usage,
cost and a `stop_reason`, written atomically (temp file + rename) so a reader never sees a partial
file. That is the artifact a scorer or an RL trainer consumes, and `nanoagent browse` reads back.

## Development

```bash
uv venv && uv sync --extra dev
uv run pytest -q
uv run ruff check .
```

## License

MIT. See [LICENSE](LICENSE).
