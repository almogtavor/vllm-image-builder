# vllm-container-builder

Build a container image with [omerpaz95's segmented spans](https://github.com/omerpaz95/vllm/pull/2) applied on top of [llm-d](https://github.com/llm-d/llm-d).

Based on the [IBM/spnl](https://github.com/IBM/spnl) build pipeline.

## How it works

1. `genpatch.sh` generates a patch from `almogtavor/vllm@segmented-spans` (cherry-picked onto the llm-d base commit)
2. `Containerfile.cuda` applies the patch on top of `ghcr.io/llm-d/llm-d-cuda:v0.5.0`
3. GitHub Actions builds and pushes to `ghcr.io/almogtavor/vllm-segmented-spans-cuda`

## Regenerating patches

```bash
cd docker/vllm/llm-d
bash genpatch.sh
```

## Building locally

```bash
docker build -f docker/vllm/llm-d/Containerfile.cuda -t vllm-segmented-spans:local .
```

## Triggering a release build

Create a GitHub release with a tag (e.g. `v0.1.0`). The workflow will build and push to GHCR.
