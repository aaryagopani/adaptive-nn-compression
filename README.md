# Edge-AI Compression Gym

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment for
compressing pretrained CIFAR-10 models. An agent interacts with the environment
one action at a time, trading model size against accuracy within a step budget.

The package and OpenEnv manifest name this project `edge-ai-compression-gym`.

## Tasks

The environment provides three tasks:

| Task | Model | Target compression | Maximum accuracy drop | Maximum steps |
| --- | --- | ---: | ---: | ---: |
| `easy` | `MiniCNN` | 1.5x | 5.0 percentage points | 15 |
| `medium` | `MiniResNet` | 2.0x | 3.0 percentage points | 20 |
| `hard` | `MiniTransformer` | 2.0x | 2.0 percentage points | 20 |

Compression ratio is baseline size divided by current size. Exceeding the
accuracy-drop limit ends the episode.

## Actions and observations

Actions are defined by `CompressionAction` in [`models.py`](./models.py):

- `quantize`: provide a layer name and `target_bits` of `4`, `8`, or `16`.
- `prune`: provide a layer name and target `sparsity` from `0.05` to `0.95`.
  This is the desired absolute sparsity, not an incremental amount.
- `fuse`: provide a `fuse_group_id` listed in the observation.
- `finish`: end the episode and trigger final scoring.

Invalid actions are rejected and still use a step. Layer names and available
fusion groups should be taken from the current observation.

[`CompressionObservation`](./models.py) includes the current model and episode
metadata, compression and accuracy progress, per-layer status, available fusion
groups, the previous action result, legal action types, and terminal status.

Quantization and fusion model the effects on size and accuracy; pruning uses
PyTorch's unstructured L1 pruning and re-evaluates accuracy. CIFAR-10 evaluation
uses a 1,000-example subset of the test set.

## Scoring

The manifest selects the `hybrid_a` step-reward function. Final scoring is
computed independently: compression achievement multiplied by accuracy
preservation, capped at `1.0`; exceeding the accuracy limit results in a score
of `0.0`. The manifest also lists `[16, 8, 16]` as recommended quantization
bits and `[0.25, 0.35]` as recommended pruning sparsities; these are guidance,
not requirements.

## Models and project files

Pretrained checkpoints in [`assets/`](./assets/) load the architectures defined
in [`server/model_architectures.py`](./server/model_architectures.py):

- `assets/mini_cnn.pth` — `MiniCNN`
- `assets/mini_resnet.pth` — `MiniResNet`
- `assets/mini_transformer.pth` — `MiniTransformer`

Other project entry points:

- [`openenv.yaml`](./openenv.yaml) — environment manifest, task settings, and
  action/observation schemas
- [`server/app.py`](./server/app.py) — FastAPI/OpenEnv server and `/health`
  endpoint
- [`server/environment.py`](./server/environment.py) — episode and compression
  logic
- [`server/graders.py`](./server/graders.py) — step rewards and final score
- [`client.py`](./client.py) — typed asynchronous OpenEnv client
- [`inference.py`](./inference.py) — LLM-driven reference agent
- [`scripts/baseline.py`](./scripts/baseline.py) — deterministic baseline agent

## Setup

Requires Python 3.10 or newer. From the repository root, create and activate a
virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, use `py -m venv .venv` and activate with
`.venv\Scripts\Activate.ps1`.

The evaluation code uses the checked-in CIFAR-10 data when available. If it
cannot use those files, it downloads CIFAR-10 and prepares a writable runtime
cache under `.runtime_data/cifar_cache/`.

## Run the environment

Start the server from the repository root:

```bash
python server/app.py
```

The server listens on `http://localhost:7860` by default. Its health endpoint
is `http://localhost:7860/health`.

To run the deterministic baseline agent, with the server running in another
terminal:

```bash
ENV_URL=http://localhost:7860 python scripts/baseline.py
```

On Windows PowerShell:

```powershell
$env:ENV_URL = "http://localhost:7860"
python scripts/baseline.py
```

## Run the LLM inference agent

`inference.py` uses the OpenAI Python client and reads the API key from
`HF_TOKEN` or `API_KEY`. Its default API base URL is
`https://api.groq.com/openai/v1`, and its default model is
`llama-3.3-70b-versatile`. Set `ENV_URL` to the endpoint you want to evaluate,
for example `https://YOUR-ENVIRONMENT.hf.space/` for a hosted environment or
`http://localhost:7860` for a local server. The default in `inference.py` is a
placeholder; set `ENV_URL` to a real endpoint before running the script.

macOS/Linux:

```bash
export HF_TOKEN="YOUR_API_KEY"
export ENV_URL="http://localhost:7860"
python inference.py
```

Windows PowerShell:

```powershell
$env:HF_TOKEN = "YOUR_API_KEY"
$env:ENV_URL = "http://localhost:7860"
python inference.py
```

Optional overrides: `API_BASE_URL` and `MODEL_NAME`. The script runs all three
tasks and emits `[START]`, `[STEP]`, and `[END]` status lines.

## Typed client example

```python
import asyncio

from client import CompressionEnvClient
from models import CompressionAction


async def main():
    async with CompressionEnvClient(base_url="http://localhost:7860") as env:
        result = await env.reset(task_id="easy")
        layer_name = next(
            layer.name for layer in result.observation.layers
            if layer.compressible and not layer.is_frozen
        )
        result = await env.step(
            CompressionAction(
                action_type="quantize",
                target_layer=layer_name,
                target_bits=16,
            )
        )
        print(result.observation.reward)


asyncio.run(main())
```
