# MLZero Claude Code Workspace

End-to-end ML automation orchestrator powered by MLZero (AutoGluon Assistant).
Claude Code skills automate the pipeline: data understanding, preprocessing,
task description, MCTS-based solution search, and result reporting.

## Quick Start

```bash
/run-mlzero dataset=/path/to/data instruction="Predict target column using regression. Metric: RMSE."
```

## Run Folder Layout

Each run creates a timestamped folder under `{{OUTPUT_ROOT}}`:

```
{{OUTPUT_ROOT}}/run_YYYYMMDD_HHMMSS/
  data/                     # Preprocessed dataset (copied/cleaned)
    train.csv
    test.csv
    ...
  mlzero_input/             # MLZero input folder (SACRED - see below)
    description.txt         # Task description (ONLY file here)
    train.csv -> ../data/   # Symlinks to data files
    test.csv  -> ../data/
    ...
  mlzero_output/            # MLZero MCTS output
    node_0/                 # First iteration
    node_1/                 # Second iteration
    ...
    best_run/               # Symlink to best node
  mlzero_config.yaml        # Config used for this run
  report.md                 # Final results report
  orchestrator_log.md       # Timestamped decision log
```

## Sacred Rule: mlzero_input/

MLZero's DataPerceptionAgent scans the input folder recursively and reads every
file it finds. The `mlzero_input/` folder must contain:
- `description.txt` (task description)
- Symlinks or copies of the actual data files

Do NOT place scripts, configs, reward functions, or any non-data files in
`mlzero_input/`. They will confuse the data perception pipeline.

## How MLZero Works

1. **Data Perception**: Scans input folder, reads file samples, understands structure
2. **Description Retrieval**: Finds and reads description.txt for task definition
3. **Task Description**: Generates precise technical task description from data + description
4. **Tool Selection**: Chooses ML tool (AutoGluon Tabular, Multimodal, TimeSeries, etc.)
5. **MCTS Search**: Iteratively generates, executes, and improves ML solutions
   - Each node generates Python code + bash execution script
   - Executes in isolated conda environments
   - Evaluates results and backpropagates scores
   - Explores different tools and approaches via tree search

## Available Tools

MLZero can select from:
- `autogluon.tabular` — tabular data (CSV, structured)
- `autogluon.multimodal` — text, images, multimodal
- `autogluon.timeseries` — time series forecasting
- `FlagEmbedding` — text embedding models
- Other registered tools

## CLI Reference

```bash
# Basic run
mlzero -i <input_folder> -o <output_folder>

# With config
mlzero -i <input_folder> -o <output_folder> -c <config.yaml>

# With instruction
mlzero -i <input_folder> -t "Predict X. Metric: Y."

# Max iterations
mlzero -i <input_folder> -n 10

# Provider selection
mlzero -i <input_folder> --provider openai
```

## Key Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `per_execution_timeout` | 28800 | Max seconds per node execution |
| `exploration_constant` | 1.414 | UCT exploration vs exploitation |
| `max_debug_depth` | 3 | Max debug chain depth |
| `initial_root_children` | 3 | Root node children (tools to try) |
| `max_debug_children` | 2 | Debug attempts per failed node |
| `max_evolve_children` | 2 | Evolution attempts per successful node |
| `score_temperature` | 0.3 | UCT score shaping temperature |

## Orchestrator Log

**Maintain `orchestrator_log.md` throughout every run.** Append timestamped
entries after every major decision:
- Dataset preparation choices
- Description content
- MLZero launch parameters
- Monitoring observations
- Issues encountered and fixes applied

## Output Path

All runs are stored under: `{{OUTPUT_ROOT}}`
