# Span Query Planning

A span query is declarative and expresses multi-generation
interactions with a model server. This design was chosen to make span
queries amenable to *planning*. This is conceptual at this point, but
we see some interesting possibilities:

- improve the quality of generated output, because map/reduce is an inference scaling technique
- increase the efficacy of attention mechanisms and KV cache locality, because the query expresses data dependencies
- allow for lightweight clients with server-managed data, because the queries express data access in a declarative way that can be managed server-side

Some progress has been made on the first point (map/reduce). See
[./haystack](the haystack experiments) for some evidence of how
map/reduce can be an inference scaling technique. However, it is as of
yet unclear what "correctness preserving" even means in the stochastic
world of LLM. Therefore deep challenges still remain.

We also have some initial experiments on the second point, via
interlinkage with [vLLM](https://github.com/vllm-project/vllm). Stay
tuned for more details.

