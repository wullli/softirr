# CUDA-enabled base matching the pinned torch==2.6.* dependency, so embedding/NLI/LLM
# measures (bipartite_embedding_distance, bipartite_nli_distance, --include-llm, ...) can use GPU.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

# meddra_graph is installed from a local path dependency (external/meddra_graph), so it
# must be present before `pip install -e .`; data/ and results/ are bind-mounted at runtime
# instead of copied in, since they're multi-hundred-MB and change independently of the code.
COPY pyproject.toml ./
COPY external/ ./external/
COPY src/ ./src/

RUN pip install -e . && \
    python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

ENTRYPOINT ["python", "-m", "soft_irr.experiments.synthetic_irr"]
