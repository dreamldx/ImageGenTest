"""Venice.ai provider (OpenAI-compatible chat completions).

Reads the API key from the ``VENICE_API_KEY`` environment variable (see
``.env``). The model defaults to Venice's own ``function_calling_default``
model but can be overridden with ``VENICE_MODEL`` — it must support function
calling, since the agent relies on tools. Select this provider with
``LLM_PROVIDER=venice``.
"""

import os

import dotenv

from .openai_compat import Provider

dotenv.load_dotenv()

VENICE_API_KEY = os.getenv("VENICE_API_KEY")

VENICE_BASE_URL = "https://api.venice.ai/api/v1"

DEFAULT_MODEL = os.getenv("VENICE_MODEL", "deepseek-v4-1-flash-1017-uncensored")

PROVIDER = Provider(
    name="venice",
    base_url=VENICE_BASE_URL,
    api_key=VENICE_API_KEY,
    api_key_env="VENICE_API_KEY",
    default_model=DEFAULT_MODEL,
    # Venice prepends its own system prompt by default; turn that off so the
    # model only sees ours (AGENT_SYSTEM_PROMPT).
    extra_body={"venice_parameters": {"include_venice_system_prompt": False}},
)
