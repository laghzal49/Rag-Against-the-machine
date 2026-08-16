# Setup and first successful run

## 1. Extract and inspect

Extract the archive and enter `rag-against-the-machine`. The root must contain
`src/`, `docs/`, `pyproject.toml`, `uv.lock`, `Makefile`, and `README.md`.
Do not run from inside `src/`: the evaluation command expects the package to be
importable from its parent directory.

Replace `YOUR_42_LOGIN` on the first README line. No login was provided in the
request, so the project does not invent one.

## 2. Install

Install `uv` using its [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
With Python 3.10 or later available:

```bash
uv sync --locked
make lint
```

`uv` creates `.venv` automatically. You do not need to activate it when using
`uv run`. `uv.lock` records dependency versions. Keep it with `pyproject.toml`.
Do not copy `.venv` between computers or include it in Git.

The mandatory environment includes Fire, Pydantic, tqdm, NumPy, Transformers,
and PyTorch. The development group includes flake8, mypy and tqdm stubs.
Only the optional semantic extra adds Sentence Transformers and its dependencies.

## 3. Place the official corpus

Copy the exact source directory from the school's attachments under `data/raw`.
For the subject's example, the result is `data/raw/vllm-0.10.1/`. Do not add an
extra wrapper directory or change the version name. Copy reference question JSON
under the corresponding `AnsweredQuestions` and `UnansweredQuestions` folders.

The small fixtures under `tests/fixtures/` are not substitutes for official
evaluation data.

## 4. Build and inspect

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How do I enable LoRA?" --k 5
```

Open the output JSON, choose one source, and verify the exact slice yourself:

```python
from pathlib import Path
from src.io_utils import read_text

text = read_text(Path("data/raw/vllm-0.10.1/docs/features/lora.md"))
# Replace start/end with numbers from an actual result.
start, end = 0, min(2000, len(text))
print(text[start:end])
```

## 5. Generate an answer

```bash
uv run python -m src answer "How do I enable LoRA?" --k 5
```

The first nonempty answer downloads `Qwen/Qwen3-0.6B` into the usual Hugging Face
cache. Model download/load latency is separate from lexical indexing and search
latency. CPU generation is supported but can be slower than retrieval by orders
of magnitude. For reproducible local tests, control CPU threads if necessary:

```bash
OMP_NUM_THREADS=4 uv run python -m src answer "How do I enable LoRA?" --k 5
```

After the model has been cached, `--local_files_only=True` disallows downloads.
A local model directory is also accepted by `--model_name`. An empty retrieval
result abstains without loading Qwen.

## 6. Prepare for submission

Review the root README's AI disclosure, run the manual checks, and add source,
configuration and documentation to your school Git repository.

Do not commit the corpus, question data, generated results, model weights or
`.venv`. `data/README.md` is the only tracked data-folder document. The evaluator
creates outputs; the commands also create their output directories as needed.
