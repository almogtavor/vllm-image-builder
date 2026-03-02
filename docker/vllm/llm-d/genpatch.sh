#!/bin/sh

set -e

SCRIPTDIR=$(cd $(dirname "$0") && pwd)

SPANS_VLLM_FORK=https://github.com/omerpaz95/vllm.git
SPANS_VLLM_BRANCH=segmented_spans
SPANS_VLLM_BASE=675ec59aa94301989c3c174b3b910338c2d51ff4

. "$SCRIPTDIR/defaults.env"

T=$(mktemp -d)
trap "rm -rf $T" EXIT

git clone $SPANS_VLLM_FORK -b $SPANS_VLLM_BRANCH $T/vllm-omerpaz95

# Notes: gzip --no-name ensures deterministic output (gzip won't save mtime in the file); this helps with git sanity
mkdir -p "$SCRIPTDIR"/patches/$LLMD_VERSION
(cd $T/vllm-omerpaz95 && git diff $SPANS_VLLM_BASE..HEAD | gzip --no-name -c > "$SCRIPTDIR"/patches/$LLMD_VERSION/01-segmented-spans.patch.gz)
