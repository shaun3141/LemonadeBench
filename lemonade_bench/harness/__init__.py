# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
LemonadeBench Evaluation Harness.

A modern CLI-based eval harness for running, comparing, and analyzing
LLM agent performance on the Lemonade Stand benchmark.

Features:
- Multi-model support (Anthropic, OpenAI, Google, and via OpenRouter)
- Multiple agent architectures (React, Plan-Act, Act-Reflect, Full)
- Goal-framing conditions for behavioral studies
- Cognitive scaffolding (calculator, code interpreter, math prompts)
- Batch runs with YAML configuration and experimental matrices
- Parallel execution with progress tracking
- Diagnostic metrics computation
- Cost tracking and estimation
- Local and cloud (Supabase) logging
"""

from .runner import Runner, RunConfig
from .config import (
    HarnessConfig,
    ModelConfig,
    LoggingConfig,
    ExperimentConfig,
    load_config,
    save_config,
    VALID_ARCHITECTURES,
    VALID_GOAL_FRAMINGS,
    VALID_TOOLS,
)
from .metrics import (
    DiagnosticMetrics,
    compute_diagnostic_metrics,
    format_metrics_summary,
)

__all__ = [
    # Runner
    "Runner",
    "RunConfig",
    # Config
    "HarnessConfig",
    "ModelConfig",
    "LoggingConfig",
    "ExperimentConfig",
    "load_config",
    "save_config",
    "VALID_ARCHITECTURES",
    "VALID_GOAL_FRAMINGS",
    "VALID_TOOLS",
    # Metrics
    "DiagnosticMetrics",
    "compute_diagnostic_metrics",
    "format_metrics_summary",
]
