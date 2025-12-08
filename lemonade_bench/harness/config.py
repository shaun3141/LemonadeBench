# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Configuration schema and loader for batch runs.

Supports YAML configuration files for reproducible benchmark runs.

Example config.yaml:
    name: "Model Comparison"
    models:
      - provider: anthropic
        name: claude-sonnet-4-20250514
        seeds: [1, 2, 3, 42, 100]
        architecture: react
        goal_framing: baseline
      - provider: openai
        name: gpt-4o
        seeds: [1, 2, 3, 42, 100]
        architecture: plan_act
        goal_framing: aggressive
    logging:
      local: true
      supabase: true
      dir: ./runs
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


# Valid architecture types
ArchitectureType = Literal["react", "plan_act", "act_reflect", "full"]
VALID_ARCHITECTURES = ["react", "plan_act", "act_reflect", "full"]

# Valid goal framing conditions
GoalFramingType = Literal["baseline", "aggressive", "conservative", "competitive", "survival", "growth"]
VALID_GOAL_FRAMINGS = ["baseline", "aggressive", "conservative", "competitive", "survival", "growth"]

# Valid scaffolding tools
VALID_TOOLS = ["calculator", "code_interpreter"]


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    provider: str
    name: str
    seeds: list[int] | None = None
    tags: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)  # e.g., ["calculator", "code_interpreter"]
    architecture: ArchitectureType = "react"  # Agent architecture: react, plan_act, act_reflect, full
    goal_framing: GoalFramingType = "baseline"  # Goal framing condition
    math_prompt: bool = False  # Enable math encouragement prompt scaffolding
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        architecture = data.get("architecture", "react")
        if architecture not in VALID_ARCHITECTURES:
            raise ValueError(f"Invalid architecture: {architecture}. Must be one of {VALID_ARCHITECTURES}")
        
        goal_framing = data.get("goal_framing", "baseline")
        if goal_framing not in VALID_GOAL_FRAMINGS:
            raise ValueError(f"Invalid goal_framing: {goal_framing}. Must be one of {VALID_GOAL_FRAMINGS}")
        
        tools = data.get("tools", [])
        for tool in tools:
            if tool not in VALID_TOOLS:
                raise ValueError(f"Invalid tool: {tool}. Must be one of {VALID_TOOLS}")
        
        return cls(
            provider=data.get("provider", "anthropic"),
            name=data["name"],
            seeds=data.get("seeds"),
            tags=data.get("tags", []),
            tools=tools,
            architecture=architecture,
            goal_framing=goal_framing,
            math_prompt=data.get("math_prompt", False),
        )


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    local: bool = True
    supabase: bool = True
    dir: str = "runs"
    
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LoggingConfig":
        if data is None:
            return cls()
        return cls(
            local=data.get("local", True),
            supabase=data.get("supabase", True),
            dir=data.get("dir", "runs"),
        )


@dataclass
class ExperimentConfig:
    """
    Configuration for running a full experimental matrix.
    
    Generates all combinations of models × seeds × goal_framings × architectures × scaffoldings.
    
    Example:
        experiment:
          provider: openrouter  # Optional: override provider for all models
          models: ["anthropic/claude-sonnet-4", "openai/gpt-4o"]
          seeds: [1, 42, 100]
          goal_framings: [baseline, aggressive, conservative]
          architectures: [react, plan_act]
          scaffoldings: [none, calculator, math_prompt]
    """
    models: list[str]  # Model specs as "provider/name"
    seeds: list[int]
    goal_framings: list[GoalFramingType] = field(default_factory=lambda: ["baseline"])
    architectures: list[ArchitectureType] = field(default_factory=lambda: ["react"])
    scaffoldings: list[str] = field(default_factory=lambda: ["none"])  # none, calculator, math_prompt, code_interpreter
    provider: str | None = None  # Optional: override provider for all models (e.g., "openrouter")
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        # Validate goal framings
        goal_framings = data.get("goal_framings", ["baseline"])
        for gf in goal_framings:
            if gf not in VALID_GOAL_FRAMINGS:
                raise ValueError(f"Invalid goal_framing: {gf}. Must be one of {VALID_GOAL_FRAMINGS}")
        
        # Validate architectures
        architectures = data.get("architectures", ["react"])
        for arch in architectures:
            if arch not in VALID_ARCHITECTURES:
                raise ValueError(f"Invalid architecture: {arch}. Must be one of {VALID_ARCHITECTURES}")
        
        # Validate scaffoldings
        scaffoldings = data.get("scaffoldings", ["none"])
        valid_scaffoldings = ["none", "calculator", "math_prompt", "code_interpreter"]
        for scaffold in scaffoldings:
            if scaffold not in valid_scaffoldings:
                raise ValueError(f"Invalid scaffolding: {scaffold}. Must be one of {valid_scaffoldings}")
        
        return cls(
            models=data.get("models", []),
            seeds=data.get("seeds", []),
            goal_framings=goal_framings,
            architectures=architectures,
            scaffoldings=scaffoldings,
            provider=data.get("provider"),  # Optional provider override
        )
    
    def generate_model_configs(self) -> list[ModelConfig]:
        """
        Generate all ModelConfig instances from the experimental matrix.
        
        Returns:
            List of ModelConfig for each combination in the matrix
        """
        configs = []
        
        for model_spec in self.models:
            # Parse provider/name from model spec
            if "/" in model_spec:
                spec_provider, name = model_spec.split("/", 1)
            else:
                # Infer provider from model name
                name = model_spec
                if "claude" in name.lower():
                    spec_provider = "anthropic"
                elif "gpt" in name.lower() or "o1" in name.lower():
                    spec_provider = "openai"
                else:
                    spec_provider = "openrouter"
            
            # Use global provider override if specified, otherwise use spec provider
            # For openrouter, the name should be the full model spec (e.g., "openai/gpt-4o-mini")
            if self.provider:
                provider = self.provider
                # If using openrouter, keep the full model spec as the name
                if self.provider == "openrouter" and "/" in model_spec:
                    name = model_spec
            else:
                provider = spec_provider
            
            for goal_framing in self.goal_framings:
                for architecture in self.architectures:
                    for scaffolding in self.scaffoldings:
                        # Convert scaffolding to tools/math_prompt
                        tools = []
                        math_prompt = False
                        
                        if scaffolding == "calculator":
                            tools = ["calculator"]
                        elif scaffolding == "code_interpreter":
                            tools = ["code_interpreter"]
                        elif scaffolding == "math_prompt":
                            math_prompt = True
                        # "none" leaves both empty
                        
                        # Create tags for this configuration
                        tags = [
                            f"goal:{goal_framing}",
                            f"arch:{architecture}",
                            f"scaffold:{scaffolding}",
                        ]
                        
                        configs.append(ModelConfig(
                            provider=provider,
                            name=name,
                            seeds=self.seeds,
                            tags=tags,
                            tools=tools,
                            architecture=architecture,
                            goal_framing=goal_framing,
                            math_prompt=math_prompt,
                        ))
        
        return configs


@dataclass
class HarnessConfig:
    """
    Complete configuration for a batch run.
    
    Supports two modes:
    1. Direct model list: Specify models directly with full configuration
    2. Experiment matrix: Use ExperimentConfig to generate combinations
    
    Attributes:
        name: Human-readable name for this batch
        models: List of model configurations to evaluate
        experiment: Optional experimental matrix configuration
        logging: Logging configuration
    """
    name: str
    models: list[ModelConfig]
    experiment: ExperimentConfig | None = None
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarnessConfig":
        # Check if using experiment matrix or direct models
        experiment = None
        models = []
        
        # Main experiment section
        if "experiment" in data:
            experiment = ExperimentConfig.from_dict(data["experiment"])
            # Generate model configs from experiment matrix
            models = experiment.generate_model_configs()
        
        # Architecture ablation section (additional study)
        if "architecture_ablation" in data:
            arch_config = ExperimentConfig.from_dict(data["architecture_ablation"])
            models.extend(arch_config.generate_model_configs())
        
        # Scaffolding ablation section (additional study)
        if "scaffolding_ablation" in data:
            scaff_config = ExperimentConfig.from_dict(data["scaffolding_ablation"])
            models.extend(scaff_config.generate_model_configs())
        
        # Direct models (fallback if no experiment sections)
        if not models and "models" in data:
            models = [ModelConfig.from_dict(m) for m in data.get("models", [])]
        
        return cls(
            name=data.get("name", "Unnamed Batch"),
            models=models,
            experiment=experiment,
            logging=LoggingConfig.from_dict(data.get("logging")),
        )
    
    def get_total_runs(self) -> int:
        """Calculate total number of runs in this configuration."""
        total = 0
        for model in self.models:
            seeds_count = len(model.seeds) if model.seeds else 1
            total += seeds_count
        return total
    
    def get_unique_runs(self) -> int:
        """
        Calculate number of unique runs after deduplication.
        
        This accounts for overlapping configs when multiple experimental
        sections share models/seeds/conditions (e.g., architecture and
        scaffolding ablations both testing baseline/react/none).
        """
        seen: set[tuple] = set()
        
        for model in self.models:
            seeds = model.seeds or [None]
            
            # Determine scaffolding type from tools/math_prompt
            if "code_interpreter" in (model.tools or []):
                scaffolding = "code_interpreter"
            elif "calculator" in (model.tools or []):
                scaffolding = "calculator"
            elif model.math_prompt:
                scaffolding = "math_prompt"
            else:
                scaffolding = "none"
            
            for seed in seeds:
                config_key = (
                    model.provider,
                    model.name,
                    seed,
                    model.goal_framing,
                    model.architecture,
                    scaffolding,
                )
                seen.add(config_key)
        
        return len(seen)


def load_config(path: str | Path) -> HarnessConfig:
    """
    Load a harness configuration from a YAML file.
    
    Args:
        path: Path to YAML config file
        
    Returns:
        HarnessConfig instance
        
    Example:
        config = load_config("config.yaml")
        for model in config.models:
            print(f"{model.provider}/{model.name}")
    """
    path = Path(path)
    
    with open(path) as f:
        data = yaml.safe_load(f)
    
    return HarnessConfig.from_dict(data)


def save_config(config: HarnessConfig, path: str | Path) -> None:
    """
    Save a harness configuration to a YAML file.
    
    Args:
        config: HarnessConfig to save
        path: Output path
    """
    path = Path(path)
    
    data: dict[str, Any] = {
        "name": config.name,
    }
    
    # If this was created from an experiment matrix, save that instead
    if config.experiment is not None:
        data["experiment"] = {
            "models": config.experiment.models,
            "seeds": config.experiment.seeds,
            "goal_framings": config.experiment.goal_framings,
            "architectures": config.experiment.architectures,
            "scaffoldings": config.experiment.scaffoldings,
        }
    else:
        data["models"] = [
            {
                "provider": m.provider,
                "name": m.name,
                "seeds": m.seeds,
                "tags": m.tags,
                "tools": m.tools,
                "architecture": m.architecture,
                "goal_framing": m.goal_framing,
                "math_prompt": m.math_prompt,
            }
            for m in config.models
        ]
    
    data["logging"] = {
        "local": config.logging.local,
        "supabase": config.logging.supabase,
        "dir": config.logging.dir,
    }
    
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Example config template
EXAMPLE_CONFIG = """# LemonadeBench Batch Configuration
name: "Model Comparison Run"

models:
  # Anthropic Claude models
  - provider: anthropic
    name: claude-sonnet-4-20250514
    seeds: [1, 2, 3, 42, 100]
    tags: [claude, sonnet]
    tools: []  # Optional tools: ["calculator", "code_interpreter"]
    architecture: react  # Options: react, plan_act, act_reflect, full
    goal_framing: baseline  # Options: baseline, aggressive, conservative, competitive, survival, growth
    math_prompt: false  # Enable math encouragement scaffolding
  
  - provider: anthropic
    name: claude-3-5-haiku-20241022
    seeds: [1, 2, 3, 42, 100]
    tags: [claude, haiku]
    tools: []
    architecture: react
    goal_framing: baseline
  
  # Example with Plan-Act architecture and aggressive framing
  # - provider: anthropic
  #   name: claude-sonnet-4-20250514
  #   seeds: [1, 2, 3, 42, 100]
  #   tags: [claude, sonnet, plan-act]
  #   tools: []
  #   architecture: plan_act
  #   goal_framing: aggressive
  
  # Example with calculator tool and math prompt
  # - provider: anthropic
  #   name: claude-sonnet-4-20250514
  #   seeds: [1, 2, 3, 42, 100]
  #   tags: [claude, sonnet, with-tools]
  #   tools: [calculator]
  #   math_prompt: true

logging:
  local: true      # Save runs to local filesystem
  supabase: true   # Also log to Supabase (if configured)
  dir: ./runs      # Local output directory

# ==========================================================================
# Available Options:
#
# Architectures (agent loop types):
#   - react: Basic observe -> decide -> act loop (default)
#   - plan_act: Generates multi-day plan before each action
#   - act_reflect: Reflects on results after each action
#   - full: Combines planning and reflection (plan -> act -> reflect)
#
# Goal Framings (motivational prompt conditions):
#   - baseline: No additional framing (default)
#   - aggressive: Risk-taking, maximize returns
#   - conservative: Loss aversion, protect capital
#   - competitive: Tournament framing, beat competitors
#   - survival: Capital preservation priority
#   - growth: Long-term learning and reputation focus
#
# Scaffolding Tools:
#   - calculator: Basic arithmetic calculations
#   - code_interpreter: Python code execution for complex analysis
#   - math_prompt: (flag) Encourage step-by-step calculations
# ==========================================================================
"""

# Example experiment matrix config
EXAMPLE_EXPERIMENT_CONFIG = """# LemonadeBench Experimental Matrix Configuration
# Use this for systematic ablation studies

name: "Goal Framing Ablation Study"

# Experimental matrix - generates all combinations
experiment:
  models:
    - anthropic/claude-sonnet-4
    - openai/gpt-4o
    - deepseek/deepseek-r1
  seeds: [1, 42, 100, 7, 2025]
  goal_framings:
    - baseline
    - aggressive
    - conservative
    - competitive
    - survival
    - growth
  architectures: [react]  # Hold architecture constant
  scaffoldings: [none]  # No additional scaffolding

logging:
  local: true
  supabase: true
  dir: ./runs

# This generates: 3 models × 5 seeds × 6 framings × 1 arch × 1 scaffold = 90 runs
"""


def create_example_config(path: str | Path = "config.yaml", experiment: bool = False) -> None:
    """
    Create an example configuration file.
    
    Args:
        path: Output path for the config file
        experiment: If True, create an experiment matrix config instead
    """
    path = Path(path)
    template = EXAMPLE_EXPERIMENT_CONFIG if experiment else EXAMPLE_CONFIG
    with open(path, "w") as f:
        f.write(template)
