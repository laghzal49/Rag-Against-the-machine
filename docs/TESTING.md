# Verification

Run `uv run python -m unittest discover -s tests -v` for error-handling and
index persistence regression checks.

Run `make lint` for application style and type checks. Follow the
[root README](../README.md#moulinette) to generate fixture predictions and
score them with the supplied moulinette. The tiny fixtures remain available
under `tests/fixtures/` for these examples.

Compare local retrieval recall with the grader using the same prediction file,
reference file, and k. See the [validation record](../VALIDATION.md) for results.

For generation, index a small corpus containing an explicit fact, ask about it,
and inspect the answer and cited passages. This requires cached or downloaded
model weights. Also check that a query without evidence abstains.

Use `make debug ARGS='search "cache" --k 5'` to inspect retrieval in pdb.
