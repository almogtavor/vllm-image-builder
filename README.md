# vllm-container-builder

Build a container image with [omerpaz95's segmented spans](https://github.com/omerpaz95/vllm/pull/2) applied on top of the official [vllm-project/vllm](https://github.com/vllm-project/vllm).

## How it works

1. Starts from the official `nvidia/cuda` base image (same approach as vllm-project/vllm's Dockerfile)
2. Clones [`almogtavor/vllm@segmented-spans`](https://github.com/almogtavor/vllm/tree/segmented-spans) — omerpaz95's feature commits rebased onto a stable upstream vllm commit
3. Installs vLLM with precompiled CUDA kernels (no compilation needed since segmented spans changes are Python-only)
4. GitHub Actions builds and pushes to `ghcr.io/almogtavor/vllm-segmented-spans-cuda`

## Configuration

Edit `docker/vllm-version` to change the vllm fork, branch, or base commit.

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
