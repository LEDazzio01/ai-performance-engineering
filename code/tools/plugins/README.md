## aipe plugins

Core `aisp` CLI looks for Typer apps exposed via the `aipe.plugins` entry point group. This lets you add or remove entire command groups without modifying the core repository.

### How to publish a plugin (in another package)

In your extension package (e.g., `aipe_ext`):

```python
# aipe_ext/dashboard_plugin.py
import typer
from common.python.capabilities import register_capability

register_capability("cli.ext", provider="aipe_ext")
register_capability("dashboard", provider="aipe_ext")

app = typer.Typer(help="Dashboard + UI commands")

@app.command()
def dashboard(...):
    ...
```

And in `pyproject.toml` (or setup.cfg):

```toml
[project.entry-points."aipe.plugins"]
dashboard = "aipe_ext.dashboard_plugin:app"
```

When installed, `aisp` will auto-attach the plugin’s Typer app under the given name (`dashboard` in this example). Capabilities can be registered at import time to re-enable gated features (e.g., `bench.llm`).

### Gated capabilities

- `cli.ext` — unlocks extension categories/commands (ai/analyze/optimize/distributed/inference/training/monitor/report/profile/hf/cluster, dashboard/mcp/tui).
- `bench.llm` — enables LLM analysis/auto-patching flags in `bench`.
- Additional capabilities are free-form (`dashboard`, `optimization_search`, etc.).

If a plugin fails to load, core CLI continues to function with only the core commands. Plugins should be defensive and avoid raising during import.***
