"""
🧠 Unified LLM Client - THE SINGLE SOURCE OF TRUTH

All LLM calls in the codebase should go through this module.
DO NOT create new LLM clients elsewhere!

Usage:
    from tools.core.llm import llm_call, get_llm_status
    
    # Simple call
    response = llm_call("Explain flash attention")
    
    # With system prompt
    response = llm_call(
        prompt="Why is my kernel slow?",
        system="You are a GPU performance expert."
    )
    
    # Check status
    status = get_llm_status()
    print(f"Provider: {status['provider']}, Model: {status['model']}")

Supported backends (auto-detected from env):
    - OpenAI (OPENAI_API_KEY)
    - Anthropic (ANTHROPIC_API_KEY)
    - Ollama (OLLAMA_HOST)
    - vLLM (VLLM_API_BASE)
"""

import os
import json
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Find repo root for .env loading
CODE_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LLMConfig:
    """LLM configuration - loaded once from environment."""
    provider: str  # openai, anthropic, ollama, vllm
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Load config from environment variables."""
        # Load .env files if not already loaded
        _load_env()
        
        # Check both new and legacy env var names
        provider = (os.environ.get('PERF_LLM_PROVIDER') or 
                   os.environ.get('LLM_PROVIDER', '')).lower()
        model = (os.environ.get('PERF_LLM_MODEL') or 
                os.environ.get('ANTHROPIC_MODEL') or 
                os.environ.get('OPENAI_MODEL', ''))
        
        # If provider explicitly set, use it
        if provider and provider not in ('auto', ''):
            # Set default model for provider if not specified
            if provider == 'openai' and not model:
                model = 'gpt-4o'
            elif provider == 'anthropic' and not model:
                model = 'claude-sonnet-4-20250514'
        # Auto-detect provider from available keys
        else:
            if os.environ.get('ANTHROPIC_API_KEY'):
                provider = 'anthropic'
                model = model or 'claude-sonnet-4-20250514'
            elif os.environ.get('OPENAI_API_KEY'):
                provider = 'openai'
                model = model or 'gpt-4o'
            elif os.environ.get('OLLAMA_HOST') or _check_ollama():
                provider = 'ollama'
                model = model or 'llama3.2'
            elif os.environ.get('VLLM_API_BASE'):
                provider = 'vllm'
                model = model or 'default'
            else:
                provider = 'none'
                model = 'none'
        
        # Handle edge case where model name matches provider
        if model.lower() == provider.lower():
            if provider == 'openai':
                model = 'gpt-4o'
            elif provider == 'anthropic':
                model = 'claude-sonnet-4-20250514'
        
        return cls(
            provider=provider,
            model=model,
            api_key=os.environ.get(f'{provider.upper()}_API_KEY') or os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'),
            base_url=os.environ.get('VLLM_API_BASE') or os.environ.get('OLLAMA_HOST'),
            temperature=float(os.environ.get('PERF_LLM_TEMPERATURE', '0.7')),
            max_tokens=int(os.environ.get('PERF_LLM_MAX_TOKENS', '4096')),
        )


def _load_env():
    """Load .env files (idempotent)."""
    for env_name in [".env", ".env.local"]:
        env_file = CODE_ROOT / env_name
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, _, value = line.partition('=')
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if env_name == ".env.local" or key not in os.environ:
                            if key and value:
                                os.environ[key] = value


def _check_ollama() -> bool:
    """Check if Ollama is running locally."""
    try:
        import urllib.request
        url = os.environ.get('OLLAMA_HOST', 'http://localhost:11434') + '/api/tags'
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False


# =============================================================================
# SINGLETON CONFIG
# =============================================================================

_config: Optional[LLMConfig] = None

def get_config() -> LLMConfig:
    """Get the singleton LLM config."""
    global _config
    if _config is None:
        _config = LLMConfig.from_env()
    return _config


def reset_config():
    """Reset config (useful for testing)."""
    global _config
    _config = None


# =============================================================================
# LLM CALL - THE SINGLE ENTRY POINT
# =============================================================================

def llm_call(
    prompt: str,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    json_mode: bool = False,
) -> str:
    """
    Make an LLM call. This is THE function all code should use.
    
    Args:
        prompt: The user prompt
        system: Optional system prompt
        temperature: Override default temperature
        max_tokens: Override default max_tokens
        json_mode: Request JSON output (OpenAI/Anthropic)
    
    Returns:
        The LLM response text
    
    Raises:
        RuntimeError: If no LLM backend is available
    """
    config = get_config()
    
    if config.provider == 'none':
        raise RuntimeError("No LLM backend configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or configure Ollama.")
    
    temp = temperature if temperature is not None else config.temperature
    tokens = max_tokens if max_tokens is not None else config.max_tokens
    
    if config.provider == 'openai':
        return _call_openai(prompt, system, temp, tokens, json_mode, config)
    elif config.provider == 'anthropic':
        return _call_anthropic(prompt, system, temp, tokens, config)
    elif config.provider == 'ollama':
        return _call_ollama(prompt, system, temp, tokens, config)
    elif config.provider == 'vllm':
        return _call_vllm(prompt, system, temp, tokens, config)
    else:
        raise RuntimeError(f"Unknown LLM provider: {config.provider}")


def _call_openai(prompt: str, system: Optional[str], temperature: float, 
                 max_tokens: int, json_mode: bool, config: LLMConfig) -> str:
    """Call OpenAI API."""
    try:
        import openai
        client = openai.OpenAI(api_key=config.api_key)
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
        
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}")


def _call_anthropic(prompt: str, system: Optional[str], temperature: float,
                    max_tokens: int, config: LLMConfig) -> str:
    """Call Anthropic API."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.api_key)
        
        kwargs = {
            "model": config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if system:
            kwargs["system"] = system
        
        response = client.messages.create(**kwargs)
        return response.content[0].text
        
    except Exception as e:
        raise RuntimeError(f"Anthropic API error: {e}")


def _call_ollama(prompt: str, system: Optional[str], temperature: float,
                 max_tokens: int, config: LLMConfig) -> str:
    """Call Ollama API."""
    import urllib.request
    
    base_url = config.base_url or 'http://localhost:11434'
    url = f"{base_url}/api/generate"
    
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    
    data = json.dumps({
        "model": config.model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }).encode('utf-8')
    
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            return result.get('response', '')
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}")


def _call_vllm(prompt: str, system: Optional[str], temperature: float,
               max_tokens: int, config: LLMConfig) -> str:
    """Call vLLM OpenAI-compatible API."""
    import urllib.request
    
    base_url = config.base_url or 'http://localhost:8000'
    url = f"{base_url}/v1/chat/completions"
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    data = json.dumps({
        "model": config.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode('utf-8')
    
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            return result['choices'][0]['message']['content']
    except Exception as e:
        raise RuntimeError(f"vLLM API error: {e}")


# =============================================================================
# STATUS & UTILITIES
# =============================================================================

def get_llm_status() -> Dict[str, Any]:
    """Get LLM backend status."""
    config = get_config()
    
    available = config.provider != 'none'
    
    # Test connection if configured
    if available:
        try:
            # Quick test with minimal tokens
            llm_call("Say 'ok'", max_tokens=10)
            available = True
        except Exception:
            available = False
    
    return {
        "available": available,
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
    }


def is_available() -> bool:
    """Quick check if LLM is available."""
    config = get_config()
    return config.provider != 'none'


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

PERF_EXPERT_SYSTEM = """You are an expert GPU performance engineer specializing in:
- CUDA kernel optimization
- PyTorch and deep learning frameworks
- Distributed training (FSDP, tensor/pipeline parallelism)
- LLM inference optimization (vLLM, TensorRT-LLM)
- Memory optimization (gradient checkpointing, mixed precision)

Provide concise, actionable advice with code examples when helpful.
Reference specific techniques from the AI Performance Engineering book when relevant."""


def ask_performance_question(question: str, context: Optional[str] = None) -> str:
    """Ask a GPU performance question with expert system prompt."""
    prompt = question
    if context:
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
    
    return llm_call(prompt, system=PERF_EXPERT_SYSTEM)


def explain_concept(concept: str) -> str:
    """Explain a GPU/AI performance concept."""
    system = PERF_EXPERT_SYSTEM + "\n\nExplain concepts clearly with: what it is, when to use it, key parameters, common pitfalls, and a code example."
    return llm_call(f"Explain: {concept}", system=system)


def analyze_code(code: str, goal: str = "optimize") -> str:
    """Analyze code for performance improvements."""
    prompt = f"""Analyze this code and suggest {goal} improvements:

```python
{code}
```

Provide specific, actionable recommendations with code examples."""
    
    return llm_call(prompt, system=PERF_EXPERT_SYSTEM)

