# Documentation: start here

This folder explains the submitted source, its algorithms, how to run it, what
was tested, and what still needs the official school attachments.

## Recommended learning order

| Step | Document | What you should understand afterward |
| --- | --- | --- |
| 1 | [SETUP.md](SETUP.md) | Install, place the corpus, run a first query |
| 2 | [STUDENT_GUIDE.md](STUDENT_GUIDE.md) | RAG, offsets, BM25, recall and defense exercises |
| 3 | [ARCHITECTURE.md](ARCHITECTURE.md) | The modules and stage-to-stage data flow |
| 4 | [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) | What every source module and function does |
| 5 | [ALGORITHMS.md](ALGORITHMS.md) | Numeric BM25, interval recall, token budgets and fusion |
| 6 | [DATA_MODELS.md](DATA_MODELS.md) | All required and internal schemas |
| 7 | [CLI.md](CLI.md) | Every command, default, output and edge case |
| 8 | [BONUSES.md](BONUSES.md) | CPU embeddings, hybrid retrieval, caching and HTTP |
| 9 | [TESTING.md](TESTING.md) | Run manual verification |
| 10 | [VALIDATION.md](VALIDATION.md) | Actual checks versus unverified official targets |
| 11 | [BENCHMARKING.md](BENCHMARKING.md) | Measure official recall and chunk-size tradeoffs |
| 12 | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Diagnose errors by pipeline stage |
| 13 | [REQUIREMENTS.md](REQUIREMENTS.md) | Trace the PDF requirements to implementation |

The root [README.md](../README.md) is the concise project overview. The student
guide includes a short Darija explanation; the main documentation is English,
as required for the project README.

## How to study effectively

Open one implementation file next to its walkthrough section. Run its smallest
example, predict an output, then check that prediction. After you understand a
function, change a parameter and explain the effect before running it. Finish
with the recode exercises and a peer review. Documentation is a map; the source
is the executable reference.

No official performance numbers are invented. The validation report separates
small local smoke tests from the corpus-wide school requirements.
