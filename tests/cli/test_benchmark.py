from __future__ import annotations

import json
import sys
from pathlib import Path

from nanoagent.cli import benchmark as cli


def test_benchmark_cli_runs_named_profile_and_benchmark(tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text('{"id":"one","input":"question","expected":"answer"}\n')
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import json, sys\n"
        "payload = json.loads(sys.stdin.readline())\n"
        "assert payload['protocol'] == 'nanoagent.runner.v1'\n"
        "print(json.dumps({'type':'done','answer':'answer','stop_reason':'answer',"
        "'steps':1,'usage':{'total_tokens':2},'cost':0.0,'error':None}))\n"
    )
    output = tmp_path / "output"
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        f"""\
model:
  model: base
  backend: sglang
  base_url: null
  api_key: null
  temperature: 0.0
  max_tokens: 16
  request_timeout: 10.0
  max_retries: 0
  extra_body: {{}}
  input_price: 0.0
  output_price: 0.0
agent:
  system_prompt: test
  max_steps: 2
  cost_limit: null
  token_limit: null
  context_window: null
  hooks: []
  skills: null
  context_files: []
  events: null
tools: []
tools_dir: null
allowed_tools: []
profiles:
  custom:
    label: Custom
    model: test-model
    harness:
      type: custom
      command: ["{sys.executable}", "{runner}"]
      cwd: null
      options: {{capabilities: {{usage: true}}}}
    model_overrides: {{}}
benchmarks:
  tiny:
    label: Tiny
    version: "1"
    primary_metric: accuracy
    adapter:
      type: jsonl_exact_match
      code: null
      options: {{path: "{dataset}"}}
profile: custom
benchmark: tiny
output: "{output}"
concurrency: 1
filter: ""
slice: ""
shuffle: false
redo: false
timeout: 10.0
"""
    )

    assert cli.main(["list", f"benchmark_cfg={config}"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["benchmarks"][0]["name"] == "tiny"
    assert listed["profiles"][0]["id"] == "custom"

    assert cli.main(["run", f"benchmark_cfg={config}"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["primary_metric"] == {"name": "accuracy", "value": 1.0}
    assert summary["profile"]["id"] == "custom"
    assert summary["profile"]["model"] == "test-model"
    assert summary["profile"]["harness"] == "custom"
    assert len(summary["profile"]["configuration_fingerprint"]) == 64
    assert '"value": 1.0' in capsys.readouterr().out
