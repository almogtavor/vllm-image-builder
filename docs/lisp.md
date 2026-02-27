## Span Queries with Rust Macros

Rust has a powerful macro system. Rust macros allow one to statically
compile complex expression trees, directly into an executable. This
may be helpful with span queries. For example, using this system, one
could generate compiled versions for each of a set of queries. These
"query executables" could then be distributed to team members, who
could then execute the queries without having to worry about
prerequisites and the like.

To explore whether this could help with the developer experience of
span queries, we have an experimental DSL. It is trivial to write a
Rust DSL using a Lisp-like syntax (of "s-expressions"), and thus we
have started there.

- `(g model input)`: Used to ask a model to generate new output.
- `(plus d1 d2 ...)`: Used to signify that the given items `d1`, `d2`,
  ... are to be considered independent of eachother.
- `(cross d1 d2 ...)`: Ibid, except now the items are to be considered
  as having a linear dependence on eachother.

### Helpers: ask, file, repeat, format
In addition to these three core operators, the DSL offers some helpful
syntactic sugarings. These include `(ask message)` which prompts the
user for a message, `(file filepath)` which reads in a string from the
given local file path, and `(repeat n <subquery>)` which expands the
given subquery `n` times.

### Examples:

This will generate (`g`) some output, using the given "model server/model", provided with the given input "Hello world":
```lisp
(g "ollama/granite3.2:2b" "Hello world")
```

Same, except ask the user (or read from a file) which prompt should be send to the generation.
```lisp
(g "ollama/granite3.2:2b" (read "What should I ask the model?"))
(g "ollama/granite3.2:2b" (file "./prompt.txt"))
```

Send a sequence of prompts to the model:
```lisp
(g "ollama/granite3.2:2b" (cross (read "What should I ask the model?")  (file "./prompt.txt")))
```

The `g` operator also accepts optional max tokens and temperature
options. Here analyze three independent inputs, each generated with
max tokens of 1000 and a temperature of 0.3:
```lisp
(g "ollama/granite3.2:2b"
   (cross (user "Pick the best one")
          (plus (g "ollama/granite3.2:2b" (user "Generate a fun email") 1000 0.3)
                (g "ollama/granite3.2:2b" (user "Generate a fun email") 1000 0.3)
                (g "ollama/granite3.2:2b" (user "Generate a fun email") 1000 0.3))))
```

Let us know what you think!
