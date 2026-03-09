# vllm-image-builder

Build a custom vLLM container image from any vLLM fork/branch. Point it at a repo and branch, and it produces a ready-to-run inference server image with precompiled CUDA kernels - no GPU compilation needed.

## What does the output image contain?

A vLLM inference server exposing the standard OpenAI-compatible API (`vllm.entrypoints.openai.api_server`), built from whichever vLLM fork and branch you configure.

## How the build works

1. Starts from `nvidia/cuda:12.4.1-devel-ubuntu22.04` (same base as vllm-project/vllm's own Dockerfile)
2. Clones the configured vLLM fork and branch
3. Installs PyTorch and vLLM dependencies
4. Installs vLLM with **precompiled CUDA kernels** from `wheels.vllm.ai` for a matching upstream commit - this works when your fork's changes are Python-only (no custom CUDA kernels)
5. GitHub Actions builds and pushes to `ghcr.io/almogtavor/vllm-segmented-spans-cuda`

## Configuration

Edit [`docker/vllm-version`](docker/vllm-version) to set:
- `VLLM_REPO` - the vLLM fork to clone
- `VLLM_BRANCH` - the branch to build
- `VLLM_BASE_COMMIT` - the upstream `vllm-project/vllm` commit your branch is based on (used to fetch matching precompiled CUDA kernels)

## Building locally

```bash
source docker/vllm-version
docker build \
  -f docker/Dockerfile.cuda \
  --build-arg VLLM_REPO=$VLLM_REPO \
  --build-arg VLLM_BRANCH=$VLLM_BRANCH \
  --build-arg VLLM_BASE_COMMIT=$VLLM_BASE_COMMIT \
  -t vllm-custom:local .
```

## Triggering a release build

Use **Actions → workflow_dispatch** with a version tag (e.g. `v0.2.0`), or create a GitHub release.

![Workflow run instructions](assets/workflow_run_instructions.png)

## Spans middleware

A lightweight Python sidecar that transforms chat completion requests into span-optimized token prompts before forwarding to vLLM.

**Image:** `ghcr.io/almogtavor/vllm-spans-middleware:<version>`

### Building the middleware image

Use **Actions → middleware image build → workflow_dispatch** with a version tag.

### Deploying on OpenShift

```bash
oc apply -n $USER-ns -f - <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spans-middleware
spec:
  replicas: 1
  selector:
    matchLabels:
      app: spans-middleware
  template:
    metadata:
      labels:
        app: spans-middleware
    spec:
      containers:
        - name: middleware
          image: ghcr.io/almogtavor/vllm-spans-middleware:<version>
          ports:
            - containerPort: 8080
          env:
            - name: SPAN_MODE
              value: "spans"
---
apiVersion: v1
kind: Service
metadata:
  name: spans-middleware
spec:
  selector:
    app: spans-middleware
  ports:
    - port: 8080
      targetPort: 8080
---
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: spans-middleware
spec:
  to:
    kind: Service
    name: spans-middleware
  port:
    targetPort: 8080
  tls:
    termination: edge
    insecureEdgeTerminationPolicy: Redirect
EOF
```

### Testing the middleware

```bash
HOST=$(oc get route spans-middleware -n $USER-ns -o jsonpath='{.spec.host}')

curl -sk https://$HOST/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
  }'
```

> **Note:** The middleware expects vLLM at `http://0.0.0.0:8000/v1` — deploy it as a sidecar in the same pod as vLLM, or modify the source to make the URL configurable.
