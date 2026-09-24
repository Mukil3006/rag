"""Central configuration. Paths work from the project root or any other directory."""
from dataclasses import dataclass, asdict
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    retrieval_top_k: int = int(os.getenv('RETRIEVAL_TOP_K', '10'))
    rerank_top_k: int = int(os.getenv('RERANK_TOP_K', '5'))
    comparison_query_focus: bool = os.getenv('COMPARISON_QUERY_FOCUS','1')=='1'
    embedding_model: str = os.getenv('EMBEDDING_MODEL', str(ROOT/'models/bge-m3'))
    reranker_model: str = os.getenv('RERANKER_MODEL', str(ROOT/'models/reranker'))
    embedding_device: str = os.getenv('EMBEDDING_DEVICE', 'cpu')
    cpu_threads: int = int(os.getenv('CPU_THREADS', '6'))
    slm_provider: str = os.getenv('SLM_PROVIDER', 'local')
    slm_model: str = os.getenv('SLM_MODEL', 'qwen3-4b-instruct-2507-q4_k_m')
    slm_url: str = os.getenv('SLM_URL', 'http://127.0.0.1:8089/v1')
    local_gguf_path: str = os.getenv('SLM_GGUF_PATH', str(ROOT/'models/verifier/Qwen3-4B-Instruct-2507-Q4_K_M.gguf'))
    local_context_tokens: int = int(os.getenv('LOCAL_CONTEXT_TOKENS','8192'))
    local_gpu_layers: int = int(os.getenv('LOCAL_GPU_LAYERS','99'))
    llm_provider: str = os.getenv('LLM_PROVIDER', 'local')
    llm_model: str = os.getenv('LLM_MODEL', 'qwen3-4b-instruct-2507-q4_k_m')
    llm_url: str = os.getenv('LLM_URL', 'http://127.0.0.1:8089/v1')
    request_timeout: int = int(os.getenv('MODEL_TIMEOUT', '180'))
    max_tokens: int = int(os.getenv('MODEL_MAX_TOKENS', '700'))
    seed: int = int(os.getenv('MODEL_SEED', '42'))
    max_parent_expansions: int = int(os.getenv('MAX_PARENT_EXPANSIONS', '5'))
    verification_retries: int = int(os.getenv('VERIFICATION_RETRIES', '1'))
    web_ui_port: int = int(os.getenv('WEB_UI_PORT', '8501'))
    local_startup_timeout: int = int(os.getenv('LOCAL_STARTUP_TIMEOUT', '120'))
    open_browser: bool = os.getenv('APP_OPEN_BROWSER', '1')=='1'

    def __post_init__(self):
        if not 1 <= self.rerank_top_k <= self.retrieval_top_k:
            raise ValueError('Require 1 <= RERANK_TOP_K <= RETRIEVAL_TOP_K')
        if self.slm_provider not in ('local','ollama'):
            raise ValueError('The verifier must use a real local model (local or ollama).')
        if self.llm_provider not in ('local','ollama','openai-compatible','mock'):
            raise ValueError('Unknown LLM_PROVIDER')
        if not 0 <= self.verification_retries <= 3:
            raise ValueError('VERIFICATION_RETRIES must be between 0 and 3')

    def public(self):
        return asdict(self)

SETTINGS = Settings()
