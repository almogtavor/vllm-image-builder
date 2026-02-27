# Feature Flags of the `spnl` library for Span Queries

> [!NOTE]
> Rust supports conditional compilation via
> ["features"](https://doc.rust-lang.org/cargo/reference/features.html). The
> `spnl` library uses this to allow you to selectively compile in more
> complex, expensive, or exploratory capabilities.

On top of these core concepts is a set of feature flags that extend
what can be expressed in a span query. For example, you may wish to
have client-side support for extended features, such as reading in
messages from a filesystem or from stdin. Or you may wish to have your
server side also support fetching message content from a
filesystem. The choice is yours.

- **rag**: This allows a span query to express that a given message
  should be augmented with fragments from a given set of
  documents. The query process, with this feature flag enabled,
  handles the fragmentation, indexing, etc.
  
- **run**: This allows for execution of a query. Without this flag
  enabled, the compiled code will only be able to parse

- **ollama**: This allows the query execution to direct `g` (generate)
  at a local Ollama model server.
  
- **openai**: This allows the query execution to direct `g` (generate)
  at an OpenAI compatible model server. By default, this will talk to
  `http://localhost:8000`, but this can be changed via the
  `OPENAI_BASE_URL` environment variable.
  
- **pull**: This allows the query execution to pull down Ollama models
  specified in a query.
  
- **tok**: This adds an API for both parsing and then tokenizing the
  messages in a query.
  
- **python_bindings**: This adds python bindings to the span query
  APIs (currently only the tokenization APIs are supported).
  
- **lisp**: A highly experimental effort to allow for [static
  compilation](./lisp) of a query into a shrinkwrapped executable.
