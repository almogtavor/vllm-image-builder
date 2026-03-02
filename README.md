# vllm-image-builder

Builds a container image of [vLLM](https://github.com/vllm-project/vllm) with [omerpaz95's segmented spans](https://github.com/omerpaz95/vllm/pull/2) feature applied on top.

## What is this image?

The output image (`ghcr.io/almogtavor/vllm-segmented-spans-cuda`) is a ready-to-run vLLM inference server that includes the **segmented spans** KV-cache optimization. It exposes the standard OpenAI-compatible API via `vllm.entrypoints.openai.api_server`.

## How the build works

1. Starts from the official `nvidia/cuda:12.4.1-devel-ubuntu22.04` base image (same approach as vllm-project/vllm's own Dockerfile)
2. Clones [`almogtavor/vllm@segmented-spans`](https://github.com/almogtavor/vllm/tree/segmented-spans) — omerpaz95's segmented-spans commits cherry-picked onto a stable upstream vLLM commit (`d7de043`)
3. Installs PyTorch and vLLM dependencies
4. Installs vLLM with **precompiled CUDA kernels** downloaded from `wheels.vllm.ai` for the matching upstream base commit — no CUDA compilation needed since the segmented spans changes are Python-only
5. GitHub Actions builds and pushes to `ghcr.io/almogtavor/vllm-segmented-spans-cuda`

## Configuration

Edit [`docker/vllm-version`](docker/vllm-version) to change the vLLM fork, branch, or base commit.

## Building locally

```bash
source docker/vllm-version
docker build \
  -f docker/Dockerfile.cuda \
  --build-arg VLLM_REPO=$VLLM_REPO \
  --build-arg VLLM_BRANCH=$VLLM_BRANCH \
  --build-arg VLLM_BASE_COMMIT=$VLLM_BASE_COMMIT \
  -t vllm-segmented-spans:local .
```

## Triggering a release build

Use **Actions → workflow_dispatch** with a version tag (e.g. `v0.2.0`), or create a GitHub release.
