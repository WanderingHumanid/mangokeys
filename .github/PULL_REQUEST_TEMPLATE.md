## What this change does

One paragraph. What is different after this pull request.

## Why

What was broken, or what was missing. Link the issue if there is one.

## How it was measured

If the change touches the model, the ranking or the data, say what was run
and what the numbers are before and after. A number that was not measured is
not a result, and the honest answer "this was not measured" is better than a
guess.

## Checklist

- `python -m pytest tests -q` passes
- `cd runtime && ./gradlew test` passes, if the Kotlin runtime changed
- The model bundle in `assets/release` was rebuilt, if the model changed
- The documentation was updated, if the behaviour changed
