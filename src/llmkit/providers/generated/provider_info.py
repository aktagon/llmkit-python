#

from __future__ import annotations

import builtins
from dataclasses import dataclass

from .providers import ProviderName


@dataclass(frozen=True)
class ProviderInfo:
    """


"""

    id: ProviderName  # typed identity (ADR-040)
    slug: str  # the slug; equal to id.value
    env_var: str
    default_model: str
    base_url: str
    #
    browser_callable: bool


_PROVIDER_INFO: dict[ProviderName, ProviderInfo] = {
    ProviderName.AI21: ProviderInfo(
        id=ProviderName.AI21,
        slug="ai21",
        env_var="AI21_API_KEY",
        default_model="jamba-1.5-large",
        base_url="https://api.ai21.com",
        browser_callable=False,
    ),
    ProviderName.ANTHROPIC: ProviderInfo(
        id=ProviderName.ANTHROPIC,
        slug="anthropic",
        env_var="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-4-6",
        base_url="https://api.anthropic.com",
        browser_callable=False,
    ),
    ProviderName.ASSEMBLYAI: ProviderInfo(
        id=ProviderName.ASSEMBLYAI,
        slug="assemblyai",
        env_var="ASSEMBLYAI_API_KEY",
        default_model="best",
        base_url="https://api.assemblyai.com",
        browser_callable=False,
    ),
    ProviderName.AZURE: ProviderInfo(
        id=ProviderName.AZURE,
        slug="azure",
        env_var="AZURE_OPENAI_API_KEY",
        default_model="gpt-4o",
        base_url="https://REPLACE-WITH-YOUR-RESOURCE.openai.azure.com",
        browser_callable=False,
    ),
    ProviderName.BEDROCK: ProviderInfo(
        id=ProviderName.BEDROCK,
        slug="bedrock",
        env_var="AWS_ACCESS_KEY_ID",
        default_model="anthropic.claude-sonnet-4-20250514-v1:0",
        base_url="https://bedrock-runtime.{region}.amazonaws.com",
        browser_callable=False,
    ),
    ProviderName.CEREBRAS: ProviderInfo(
        id=ProviderName.CEREBRAS,
        slug="cerebras",
        env_var="CEREBRAS_API_KEY",
        default_model="llama-3.3-70b",
        base_url="https://api.cerebras.ai",
        browser_callable=False,
    ),
    ProviderName.COHERE: ProviderInfo(
        id=ProviderName.COHERE,
        slug="cohere",
        env_var="COHERE_API_KEY",
        default_model="command-r-plus",
        base_url="https://api.cohere.com/compatibility",
        browser_callable=False,
    ),
    ProviderName.DEEPSEEK: ProviderInfo(
        id=ProviderName.DEEPSEEK,
        slug="deepseek",
        env_var="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        base_url="https://api.deepseek.com",
        browser_callable=False,
    ),
    ProviderName.DOUBAO: ProviderInfo(
        id=ProviderName.DOUBAO,
        slug="doubao",
        env_var="ARK_API_KEY",
        default_model="doubao-1.5-pro-32k-250115",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        browser_callable=False,
    ),
    ProviderName.ERNIE: ProviderInfo(
        id=ProviderName.ERNIE,
        slug="ernie",
        env_var="QIANFAN_API_KEY",
        default_model="ernie-4.0-8k",
        base_url="https://qianfan.baidubce.com/v2",
        browser_callable=False,
    ),
    ProviderName.FIREWORKS: ProviderInfo(
        id=ProviderName.FIREWORKS,
        slug="fireworks",
        env_var="FIREWORKS_API_KEY",
        default_model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        base_url="https://api.fireworks.ai/inference",
        browser_callable=False,
    ),
    ProviderName.GOOGLE: ProviderInfo(
        id=ProviderName.GOOGLE,
        slug="google",
        env_var="GOOGLE_API_KEY",
        default_model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com",
        browser_callable=True,
    ),
    ProviderName.GROK: ProviderInfo(
        id=ProviderName.GROK,
        slug="grok",
        env_var="XAI_API_KEY",
        default_model="grok-3-fast",
        base_url="https://api.x.ai",
        browser_callable=False,
    ),
    ProviderName.GROQ: ProviderInfo(
        id=ProviderName.GROQ,
        slug="groq",
        env_var="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai",
        browser_callable=False,
    ),
    ProviderName.INWORLD: ProviderInfo(
        id=ProviderName.INWORLD,
        slug="inworld",
        env_var="INWORLD_API_KEY",
        default_model="inworld-tts-2",
        base_url="https://api.inworld.ai",
        browser_callable=False,
    ),
    ProviderName.JAN: ProviderInfo(
        id=ProviderName.JAN,
        slug="jan",
        env_var="JAN_API_KEY",
        default_model="",
        base_url="http://localhost:1337",
        browser_callable=False,
    ),
    ProviderName.LLAMACPP: ProviderInfo(
        id=ProviderName.LLAMACPP,
        slug="llamacpp",
        env_var="LLAMACPP_API_KEY",
        default_model="",
        base_url="http://localhost:8080",
        browser_callable=False,
    ),
    ProviderName.LMSTUDIO: ProviderInfo(
        id=ProviderName.LMSTUDIO,
        slug="lmstudio",
        env_var="LM_STUDIO_API_KEY",
        default_model="",
        base_url="http://localhost:1234",
        browser_callable=False,
    ),
    ProviderName.MINIMAX: ProviderInfo(
        id=ProviderName.MINIMAX,
        slug="minimax",
        env_var="MINIMAX_API_KEY",
        default_model="MiniMax-Text-01",
        base_url="https://api.minimax.chat",
        browser_callable=False,
    ),
    ProviderName.MISTRAL: ProviderInfo(
        id=ProviderName.MISTRAL,
        slug="mistral",
        env_var="MISTRAL_API_KEY",
        default_model="mistral-large-latest",
        base_url="https://api.mistral.ai",
        browser_callable=False,
    ),
    ProviderName.MOONSHOT: ProviderInfo(
        id=ProviderName.MOONSHOT,
        slug="moonshot",
        env_var="MOONSHOT_API_KEY",
        default_model="moonshot-v1-8k",
        base_url="https://api.moonshot.ai",
        browser_callable=False,
    ),
    ProviderName.OLLAMA: ProviderInfo(
        id=ProviderName.OLLAMA,
        slug="ollama",
        env_var="OLLAMA_API_KEY",
        default_model="",
        base_url="http://localhost:11434",
        browser_callable=False,
    ),
    ProviderName.OPENAI: ProviderInfo(
        id=ProviderName.OPENAI,
        slug="openai",
        env_var="OPENAI_API_KEY",
        default_model="gpt-4o-2024-08-06",
        base_url="https://api.openai.com",
        browser_callable=False,
    ),
    ProviderName.OPENROUTER: ProviderInfo(
        id=ProviderName.OPENROUTER,
        slug="openrouter",
        env_var="OPENROUTER_API_KEY",
        default_model="openai/gpt-4o",
        base_url="https://openrouter.ai/api",
        browser_callable=False,
    ),
    ProviderName.PERPLEXITY: ProviderInfo(
        id=ProviderName.PERPLEXITY,
        slug="perplexity",
        env_var="PERPLEXITY_API_KEY",
        default_model="sonar-pro",
        base_url="https://api.perplexity.ai",
        browser_callable=False,
    ),
    ProviderName.PIXVERSE: ProviderInfo(
        id=ProviderName.PIXVERSE,
        slug="pixverse",
        env_var="PIXVERSE_API_KEY",
        default_model="v4.5",
        base_url="https://app-api.pixverse.ai",
        browser_callable=False,
    ),
    ProviderName.QWEN: ProviderInfo(
        id=ProviderName.QWEN,
        slug="qwen",
        env_var="DASHSCOPE_API_KEY",
        default_model="qwen-plus",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode",
        browser_callable=False,
    ),
    ProviderName.RECRAFT: ProviderInfo(
        id=ProviderName.RECRAFT,
        slug="recraft",
        env_var="RECRAFT_API_TOKEN",
        default_model="recraftv3",
        base_url="https://external.api.recraft.ai",
        browser_callable=False,
    ),
    ProviderName.SAMBANOVA: ProviderInfo(
        id=ProviderName.SAMBANOVA,
        slug="sambanova",
        env_var="SAMBANOVA_API_KEY",
        default_model="Meta-Llama-3.3-70B-Instruct",
        base_url="https://api.sambanova.ai",
        browser_callable=False,
    ),
    ProviderName.TOGETHER: ProviderInfo(
        id=ProviderName.TOGETHER,
        slug="together",
        env_var="TOGETHER_API_KEY",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        base_url="https://api.together.xyz",
        browser_callable=False,
    ),
    ProviderName.VERTEX: ProviderInfo(
        id=ProviderName.VERTEX,
        slug="vertex",
        env_var="VERTEX_BEARER_TOKEN",
        default_model="imagen-3.0-generate-002",
        base_url="https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models",
        browser_callable=False,
    ),
    ProviderName.VIDU: ProviderInfo(
        id=ProviderName.VIDU,
        slug="vidu",
        env_var="VIDU_API_KEY",
        default_model="viduq3-pro",
        base_url="https://api.vidu.com",
        browser_callable=False,
    ),
    ProviderName.VLLM: ProviderInfo(
        id=ProviderName.VLLM,
        slug="vllm",
        env_var="VLLM_API_KEY",
        default_model="",
        base_url="http://localhost:8000",
        browser_callable=False,
    ),
    ProviderName.WORKERSAI: ProviderInfo(
        id=ProviderName.WORKERSAI,
        slug="workersai",
        env_var="CLOUDFLARE_API_TOKEN",
        default_model="@cf/meta/llama-3.1-8b-instruct",
        base_url="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        browser_callable=False,
    ),
    ProviderName.YI: ProviderInfo(
        id=ProviderName.YI,
        slug="yi",
        env_var="YI_API_KEY",
        default_model="yi-large",
        base_url="https://api.01.ai",
        browser_callable=False,
    ),
    ProviderName.ZHIPU: ProviderInfo(
        id=ProviderName.ZHIPU,
        slug="zhipu",
        env_var="ZHIPU_API_KEY",
        default_model="glm-4-plus",
        base_url="https://open.bigmodel.cn/api/paas",
        browser_callable=False,
    ),
}


def info(provider: ProviderName) -> ProviderInfo:
    """
"""
    return _PROVIDER_INFO[provider]


def list() -> builtins.list[ProviderInfo]:
    """"""
    return sorted(_PROVIDER_INFO.values(), key=lambda i: i.slug)
