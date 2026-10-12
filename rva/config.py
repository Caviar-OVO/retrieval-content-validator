"""Configuration loading.

Reads settings from a `.env` file in the project root (and from real environment
variables). The file is the easiest place to keep your API key locally; it is
git-ignored by default.

Precedence (highest first):
    1. real environment variables
    2. values in the `.env` file
    3. built-in defaults

Recognised keys:
    OPENAI_API_KEY     your key (required for the ragas / trulens grounding judges)
    OPENAI_BASE_URL    optional, for a proxy or OpenAI-compatible endpoint
    OPENAI_API_BASE    alias for OPENAI_BASE_URL (some tools read this name)
    RVA_JUDGE_MODEL    model used by the LLM judge (default: gpt-4o-mini)
    RVA_EMBED_MODEL    embedding model for vector retrieval (default: text-embedding-3-small)
    RVA_JUDGE          default grounding backend: local (off) | ragas | trulens
    RVA_RETRIEVER      retrieval backend for Stage 2: hash | openai | auto
    RVA_NLI            entailment backend for Stage 2: heuristic | llm | auto
    RVA_DOMAIN         default topic domain for credibility weighting
    RVA_REVISE         1/true to produce a revised answer by default (default: 0)
    RVA_REVISER        revision backend: local (default, offline) | llm | auto
    RVA_CRITIC         critic backend: deterministic (default, offline) | llm | auto
    RVA_RECHECK       1/true to run the revision->recheck gate (default: 1)
    RVA_RECHECK_THRESHOLD  score below which the recheck gate forces another edit (default: 0.7)
    RVA_RECHECK_MAX_ROUNDS maximum revision->recheck rounds (default: 2)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

_loaded = False

# Keys the pipeline reads via os.environ; a Streamlit Cloud deployment supplies
# them through st.secrets instead of a local .env file.
_MANAGED_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "RVA_JUDGE_MODEL",
    "RVA_EMBED_MODEL",
    "RVA_JUDGE",
    "RVA_RETRIEVER",
    "RVA_NLI",
    "RVA_DOMAIN",
    "RVA_REVISE",
    "RVA_REVISER",
    "RVA_CRITIC",
    "RVA_RECHECK",
    "RVA_RECHECK_THRESHOLD",
    "RVA_RECHECK_MAX_ROUNDS",
)


def _load_streamlit_secrets() -> None:
    """Bridge Streamlit `st.secrets` into os.environ (cloud deployments).

    A no-op when Streamlit is not installed/running or no secrets are set, so
    the local command-line and test flows are unaffected. Real environment
    variables and a local `.env` always win.
    """
    try:
        import streamlit as st  # noqa: F401
        secrets = st.secrets
    except Exception:
        return

    for key in _MANAGED_KEYS:
        if key in os.environ:
            continue
        try:
            value = secrets[key]
        except Exception:
            continue
        if value is None:
            continue
        os.environ[key] = str(value).strip()


def _load_env_file(path: Path = ENV_PATH) -> None:
    """Load key=value pairs from `path` into os.environ (without overwriting)."""
    global _loaded
    if _loaded:
        return
    _loaded = True

    # In the cloud there is no .env file: pull from st.secrets first, then let
    # a real .env (if any) fill anything still missing.
    _load_streamlit_secrets()

    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except Exception:
        pass

    # Fallback: minimal parser so the project works without python-dotenv.
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key() -> Optional[str]:
    _load_env_file()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return key or None


def get_base_url() -> Optional[str]:
    _load_env_file()
    return (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "").strip() or None


def get_judge_model() -> str:
    _load_env_file()
    return os.environ.get("RVA_JUDGE_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"


def get_embed_model() -> str:
    _load_env_file()
    return os.environ.get("RVA_EMBED_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small"


def get_retriever_backend() -> str:
    _load_env_file()
    return os.environ.get("RVA_RETRIEVER", "hash").strip().lower() or "hash"


def get_nli_backend() -> str:
    _load_env_file()
    return os.environ.get("RVA_NLI", "heuristic").strip().lower() or "heuristic"


def get_revise_default() -> bool:
    _load_env_file()
    return os.environ.get("RVA_REVISE", "0").strip().lower() in ("1", "true", "yes", "on")


def get_reviser_backend() -> str:
    _load_env_file()
    return os.environ.get("RVA_REVISER", "local").strip().lower() or "local"


def get_critic_backend() -> str:
    _load_env_file()
    return os.environ.get("RVA_CRITIC", "deterministic").strip().lower() or "deterministic"


def get_recheck_enabled() -> bool:
    _load_env_file()
    return os.environ.get("RVA_RECHECK", "1").strip().lower() in ("1", "true", "yes", "on")


def get_recheck_threshold() -> float:
    _load_env_file()
    try:
        return float(os.environ.get("RVA_RECHECK_THRESHOLD", "0.7"))
    except ValueError:
        return 0.7


def get_recheck_max_rounds() -> int:
    _load_env_file()
    try:
        return max(1, int(os.environ.get("RVA_RECHECK_MAX_ROUNDS", "2")))
    except ValueError:
        return 2


def get_default_judge() -> str:
    _load_env_file()
    return os.environ.get("RVA_JUDGE", "local").strip().lower() or "local"


def get_default_domain() -> str:
    _load_env_file()
    return os.environ.get("RVA_DOMAIN", "general").strip() or "general"


def describe() -> dict:
    """A redacted summary safe to print."""
    _load_env_file()
    key = get_api_key()
    return {
        "env_file": str(ENV_PATH),
        "env_file_exists": ENV_PATH.exists(),
        "openai_api_key": ("set (" + key[:6] + "…" + key[-4:] + ")") if key and len(key) > 12 else ("set" if key else "not set"),
        "openai_base_url": get_base_url() or "(default)",
        "judge_model": get_judge_model(),
        "embed_model": get_embed_model(),
        "default_judge": get_default_judge(),
        "retriever": get_retriever_backend(),
        "nli": get_nli_backend(),
        "default_domain": get_default_domain(),
        "revise": get_revise_default(),
        "critic": get_critic_backend(),
        "reviser": get_reviser_backend(),
        "recheck": get_recheck_enabled(),
        "recheck_threshold": get_recheck_threshold(),
        "recheck_max_rounds": get_recheck_max_rounds(),
    }