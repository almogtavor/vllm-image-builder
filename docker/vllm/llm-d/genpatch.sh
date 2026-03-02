#!/bin/sh

set -e

SCRIPTDIR=$(cd $(dirname "$0") && pwd)

SPANS_VLLM_FORK=https://github.com/almogtavor/vllm.git
SPANS_VLLM_BRANCH=segmented-spans

. "$SCRIPTDIR/defaults.env"
SPANS_VLLM_BASE=$VLLM_COMMIT_SHA

T=$(mktemp -d)
trap "rm -rf $T" EXIT

git clone $SPANS_VLLM_FORK -b $SPANS_VLLM_BRANCH $T/vllm

# Notes: gzip --no-name ensures deterministic output (gzip won't save mtime in the file); this helps with git sanity
mkdir -p "$SCRIPTDIR"/patches/$LLMD_VERSION
(cd $T/vllm && git diff $SPANS_VLLM_BASE..HEAD | gzip --no-name -c > "$SCRIPTDIR"/patches/$LLMD_VERSION/01-segmented-spans.patch.gz)
