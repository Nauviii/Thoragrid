from pathlib import Path


def touch(path: Path):
    """Create file and parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def create_project():
    p = Path(".")

    # Directories
    directories = [
        "notebooks",
        "config",
        "models/weights",
        "logs",
        "data/raw",
        "data/processed",
        "data/knowledge_base",
        "core/cnn",
        "core/clip",
        "core/gradcam",
        "core/rag",
        "core/llm",
        "core/sql_agent",
        "api/routes",
        "api/schemas",
        "api/middleware",
        "app/pages",
        "app/components",
        "scripts",
        "tests/unit",
        "tests/integration",
    ]

    for d in directories:
        (p / d).mkdir(parents=True, exist_ok=True)

    # Files
    files = [
        # notebooks
        "notebooks/train_densenet121.py",
        "notebooks/gradcam_exploration.py",
        "notebooks/threshold_optimization.py",
        "notebooks/dataset_utils.py",

        # config
        "config/__init__.py",
        "config/settings.py",

        # core
        "core/__init__.py",
        "core/cnn/__init__.py",
        "core/cnn/inference.py",

        "core/clip/__init__.py",
        "core/clip/validator.py",

        "core/gradcam/__init__.py",
        "core/gradcam/explainer.py",
        "core/gradcam/region_map.py",

        "core/rag/__init__.py",
        "core/rag/ingestor.py",
        "core/rag/retriever.py",

        "core/llm/__init__.py",
        "core/llm/client.py",
        "core/llm/prompts.py",

        "core/sql_agent/__init__.py",
        "core/sql_agent/agent.py",
        "core/sql_agent/guardrails.py",

        # api
        "api/__init__.py",
        "api/main.py",

        "api/routes/__init__.py",
        "api/routes/auth.py",
        "api/routes/text.py",
        "api/routes/image.py",
        "api/routes/history.py",
        "api/routes/feedback.py",
        "api/routes/agent.py",

        "api/schemas/__init__.py",
        "api/schemas/requests.py",
        "api/schemas/responses.py",

        "api/middleware/__init__.py",
        "api/middleware/auth.py",
        "api/middleware/logging.py",

        # app
        "app/__init__.py",
        "app/main.py",

        "app/pages/__init__.py",
        "app/pages/text_qa.py",
        "app/pages/image_analysis.py",
        "app/pages/history.py",
        "app/pages/sql_agent.py",

        "app/components/__init__.py",
        "app/components/confidence_chart.py",

        # scripts
        "scripts/db_models.py",
        "scripts/db_init.py",
        "scripts/ingest_openi.py",
        "scripts/ingest_statpearls.py",

        # tests
        "tests/__init__.py",

        "tests/unit/test_cnn.py",
        "tests/unit/test_clip.py",
        "tests/unit/test_gradcam.py",
        "tests/unit/test_rag.py",
        "tests/unit/test_llm.py",

        "tests/integration/test_image_pipeline.py",
        "tests/integration/test_text_pipeline.py",

        # root
        "requirements.txt",
        "README.md",
    ]

    for f in files:
        touch(p / f)

    # .gitignore
    gitignore = """
.env
models/weights/
data/
logs/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.DS_Store
""".strip()

    (p / ".gitignore").write_text(gitignore, encoding="utf-8")

    # .env.example
    env_example = """
GROQ_API_KEY=gsk_...
JWT_SECRET_KEY=change_this_to_64char_random_string
DATABASE_URL=postgresql+psycopg2://user:password@db.supabase.co:5432/postgres
SQL_AGENT_READONLY_URL=postgresql+psycopg2://readonly:password@db.supabase.co:5432/postgres
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=medassist-knowledge
MODEL_REPO_ID=your-hf-username/medassist-densenet121
""".strip()

    (p / ".env.example").write_text(env_example, encoding="utf-8")

    print(f"Done — project created at: {p}")


if __name__ == "__main__":
    create_project()