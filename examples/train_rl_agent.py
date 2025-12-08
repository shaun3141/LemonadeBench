#!/usr/bin/env python3
# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Train an RL agent on the Lemonade Stand environment.

This script demonstrates how to train traditional RL algorithms on LemonadeBench
using Stable Baselines3 with automatic hardware optimization.

Usage:
    # Show detected hardware and recommended settings
    python examples/train_rl_agent.py --show-hardware

    # Train with default settings (auto-detects hardware)
    python examples/train_rl_agent.py

    # Train with high-performance mode (maximizes hardware utilization)
    python examples/train_rl_agent.py --high-perf

    # Train with custom algorithm and timesteps
    python examples/train_rl_agent.py --algo ppo --timesteps 500000

    # Train with RecurrentPPO (LSTM for temporal reasoning)
    python examples/train_rl_agent.py --algo recurrent_ppo --timesteps 500000

    # Force CPU-only training
    python examples/train_rl_agent.py --device cpu

    # Evaluate a trained model
    python examples/train_rl_agent.py --evaluate --model-path examples/rl_models/lemonade_ppo

Requirements:
    pip install gymnasium stable-baselines3

    Or install with the rl extras:
    pip install -e ".[rl]"

    For GPU support (recommended for faster training):
    pip install torch --index-url https://download.pytorch.org/whl/cu126
"""

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Add parent directory for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

# Default output directory for models (relative to script location)
MODELS_DIR = Path(__file__).parent / "rl_models"

try:
    import torch
    from stable_baselines3 import PPO, SAC, A2C, TD3
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.env_util import make_vec_env
except ImportError:
    print("Error: stable-baselines3 is required for training.")
    print("Install with: pip install stable-baselines3")
    sys.exit(1)

# Try to import RecurrentPPO from sb3-contrib (optional)
try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT_PPO = True
except ImportError:
    HAS_RECURRENT_PPO = False
    RecurrentPPO = None

from lemonade_bench.agents.rl import LemonadeGymEnv, make_env


# =============================================================================
# Hardware Detection and Optimization
# =============================================================================

@dataclass
class HardwareInfo:
    """System hardware information for optimization."""
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    has_cuda: bool
    cuda_version: Optional[str]
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    gpu_compute_capability: Optional[tuple]
    supports_tf32: bool  # Ampere+ (compute capability >= 8.0)
    supports_bf16: bool  # Ampere+ with good BF16 support


def detect_hardware() -> HardwareInfo:
    """Detect system hardware capabilities."""
    # CPU detection
    cpu_cores = os.cpu_count() or 4
    # On systems with SMT/HT, this returns logical cores (threads)
    # Physical cores are typically half on most consumer CPUs
    cpu_threads = cpu_cores
    cpu_cores = max(1, cpu_cores // 2)  # Estimate physical cores
    
    # RAM detection (cross-platform)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # Fallback: try to read from /proc/meminfo on Linux
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        ram_gb = int(line.split()[1]) / (1024 ** 2)
                        break
                else:
                    ram_gb = 16.0  # Conservative default
        except (FileNotFoundError, IOError):
            ram_gb = 16.0  # Conservative default
    
    # GPU detection
    has_cuda = torch.cuda.is_available()
    cuda_version = None
    gpu_name = None
    gpu_memory_gb = None
    gpu_compute_capability = None
    supports_tf32 = False
    supports_bf16 = False
    
    if has_cuda:
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        gpu_compute_capability = torch.cuda.get_device_capability(0)
        
        # Ampere+ GPUs (compute capability >= 8.0) support TF32 and BF16
        if gpu_compute_capability[0] >= 8:
            supports_tf32 = True
            supports_bf16 = True
    
    return HardwareInfo(
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        ram_gb=ram_gb,
        has_cuda=has_cuda,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        gpu_compute_capability=gpu_compute_capability,
        supports_tf32=supports_tf32,
        supports_bf16=supports_bf16,
    )


def configure_torch_optimizations(hw: HardwareInfo, verbose: bool = True):
    """
    Configure PyTorch for optimal performance based on detected hardware.
    
    Enables:
    - TensorFloat32 (TF32) on Ampere+ GPUs for faster matmul
    - cuDNN autotuning for optimal convolution algorithms
    - Proper device selection
    """
    optimizations_applied = []
    
    if hw.has_cuda:
        # Enable TF32 for Ampere+ GPUs (compute capability >= 8.0)
        # TF32 provides ~3x speedup with minimal precision loss for training
        if hw.supports_tf32:
            torch.set_float32_matmul_precision('high')  # Use TF32 for matmul
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            optimizations_applied.append("TensorFloat32 (TF32) enabled")
        
        # Enable cuDNN autotuning - finds optimal algorithms for your GPU
        # Small overhead on first run, but faster subsequent operations
        torch.backends.cudnn.benchmark = True
        optimizations_applied.append("cuDNN benchmark mode enabled")
        
        # Disable cuDNN deterministic mode for better performance
        # (Set to True if you need reproducible results at cost of speed)
        torch.backends.cudnn.deterministic = False
    
    # Set number of threads for CPU operations
    # This helps when GPU is doing forward/backward passes and CPU is doing env simulation
    torch.set_num_threads(min(hw.cpu_threads, 8))  # Cap at 8 to avoid oversubscription
    optimizations_applied.append(f"CPU threads set to {min(hw.cpu_threads, 8)}")
    
    if verbose and optimizations_applied:
        print("PyTorch Optimizations Applied:")
        for opt in optimizations_applied:
            print(f"  ✓ {opt}")
        print()
    
    return optimizations_applied


@dataclass
class PerformancePreset:
    """Hyperparameter preset for different hardware configurations."""
    name: str
    n_envs: int
    use_subproc: bool
    batch_size: int
    n_steps: int
    buffer_size: int  # For off-policy algorithms (SAC, TD3)
    net_arch_size: int  # Network hidden layer size
    lstm_hidden_size: int  # For RecurrentPPO
    learning_rate: float


# Network size presets - right-sized for the 27-dim observation space
# Larger networks don't help and may overfit on this simple environment
NET_SIZE_PRESETS = {
    "tiny": 32,      # ~3K params - minimal, fast
    "small": 64,     # ~12K params - recommended for LemonadeBench
    "medium": 128,   # ~50K params - still reasonable
    "large": 256,    # ~200K params - probably overkill
    "xlarge": 512,   # ~800K params - definitely overkill
}


def get_performance_preset(hw: HardwareInfo, high_perf: bool = False) -> PerformancePreset:
    """
    Get optimized hyperparameters based on hardware capabilities.
    
    Args:
        hw: Hardware information
        high_perf: If True, maximize hardware utilization (higher memory usage)
    
    Returns:
        PerformancePreset with optimized hyperparameters
    """
    # Base calculations
    # For vectorized envs, use physical core count (env simulation is CPU-bound)
    base_n_envs = hw.cpu_cores
    
    # Network size: Use small (64) by default - right-sized for 27-dim observations
    # Larger networks don't improve performance on this simple environment
    net_arch_size = 64
    lstm_hidden_size = 64
    
    if high_perf:
        # High-performance mode: maximize throughput (not model size)
        n_envs = min(hw.cpu_threads, 32)  # Use all threads, cap at 32
        use_subproc = True  # Always use subprocess for true parallelism
        
        # Larger batches improve GPU utilization
        if hw.has_cuda and hw.gpu_memory_gb:
            if hw.gpu_memory_gb >= 8:
                batch_size = 2048
            else:
                batch_size = 1024
        else:
            batch_size = 512 if hw.ram_gb >= 16 else 256
        
        # Larger rollout buffer for high-perf
        n_steps = 4096
        
        # Larger replay buffer for off-policy (scale with RAM)
        buffer_size = min(int(hw.ram_gb * 50_000), 2_000_000)  # ~50k per GB, cap at 2M
        
        learning_rate = 3e-4
        name = "high_performance"
        
    else:
        # Standard mode: balanced performance
        n_envs = max(4, min(base_n_envs, 16))
        use_subproc = hw.cpu_cores >= 4
        batch_size = 256
        n_steps = 2048
        buffer_size = 100_000
        learning_rate = 3e-4
        name = "standard"
    
    return PerformancePreset(
        name=name,
        n_envs=n_envs,
        use_subproc=use_subproc,
        batch_size=batch_size,
        n_steps=n_steps,
        buffer_size=buffer_size,
        net_arch_size=net_arch_size,
        lstm_hidden_size=lstm_hidden_size,
        learning_rate=learning_rate,
    )


def print_hardware_info(hw: HardwareInfo, preset: PerformancePreset):
    """Print detected hardware and selected configuration."""
    print("=" * 70)
    print("Hardware Detection & Configuration")
    print("=" * 70)
    print()
    print("Detected Hardware:")
    print(f"  CPU: {hw.cpu_cores} cores / {hw.cpu_threads} threads")
    print(f"  RAM: {hw.ram_gb:.1f} GB")
    
    if hw.has_cuda:
        print(f"  GPU: {hw.gpu_name}")
        print(f"  GPU Memory: {hw.gpu_memory_gb:.1f} GB")
        print(f"  CUDA: {hw.cuda_version}")
        print(f"  Compute Capability: {hw.gpu_compute_capability[0]}.{hw.gpu_compute_capability[1]}")
        if hw.supports_tf32:
            print("  TF32 Support: ✓ (Ampere+ GPU detected)")
    else:
        print("  GPU: Not available (CPU-only training)")
    
    print()
    print(f"Performance Preset: {preset.name.upper()}")
    print(f"  Parallel Environments: {preset.n_envs}")
    print(f"  Vectorization: {'SubprocVecEnv' if preset.use_subproc else 'DummyVecEnv'}")
    print(f"  Batch Size: {preset.batch_size}")
    print(f"  Rollout Steps: {preset.n_steps}")
    print(f"  Network Size: {preset.net_arch_size}")
    print()


# =============================================================================
# Algorithms and Training
# =============================================================================

ALGORITHMS = {
    "ppo": PPO,
    "sac": SAC,
    "a2c": A2C,
    "td3": TD3,
}

# Add RecurrentPPO if available
if HAS_RECURRENT_PPO:
    ALGORITHMS["recurrent_ppo"] = RecurrentPPO


def create_env(seed: int = 0, render_mode: str = None):
    """Create a single environment instance."""
    return LemonadeGymEnv(seed=seed, render_mode=render_mode)


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    
    Args:
        initial_value: Initial learning rate
        
    Returns:
        Schedule function that computes current learning rate
    """
    def func(progress_remaining: float) -> float:
        """
        Progress remaining goes from 1.0 (start) to 0.0 (end of training).
        """
        return progress_remaining * initial_value
    return func


def train(
    algo: str = "ppo",
    timesteps: int = 100_000,
    n_envs: int = None,  # Auto-detect if None
    save_path: str = None,
    eval_freq: int = 10_000,
    seed: int = 42,
    verbose: int = 1,
    use_subproc: bool = None,  # Auto-detect if None
    lr_schedule: str = "constant",
    device: str = "auto",  # "auto", "cuda", "cpu"
    high_perf: bool = False,
    batch_size: int = None,  # Override preset if specified
    n_steps: int = None,  # Override preset if specified
    net_size: str = "small",  # Network size preset
):
    """
    Train an RL agent on the Lemonade Stand environment.
    
    Args:
        algo: Algorithm name (ppo, sac, a2c, td3, recurrent_ppo)
        timesteps: Total training timesteps
        n_envs: Number of parallel environments (auto-detect if None)
        save_path: Path to save the trained model
        eval_freq: Evaluation frequency in timesteps
        seed: Random seed
        verbose: Verbosity level
        use_subproc: Use SubprocVecEnv for true parallelism (auto-detect if None)
        lr_schedule: Learning rate schedule ("constant" or "linear")
        device: Device for training ("auto", "cuda", "cpu")
        high_perf: Enable high-performance mode
        batch_size: Override batch size (uses preset if None)
        n_steps: Override rollout steps (uses preset if None)
        net_size: Network size preset (tiny/small/medium/large/xlarge)
    """
    if algo.lower() not in ALGORITHMS:
        print(f"Error: Unknown algorithm '{algo}'")
        print(f"Available: {list(ALGORITHMS.keys())}")
        sys.exit(1)
    
    algo_class = ALGORITHMS[algo.lower()]
    
    # Detect hardware and get optimized settings
    hw = detect_hardware()
    preset = get_performance_preset(hw, high_perf=high_perf)
    
    # Apply PyTorch optimizations
    if verbose:
        print_hardware_info(hw, preset)
    configure_torch_optimizations(hw, verbose=verbose > 0)
    
    # Resolve device
    if device == "auto":
        device = "cuda" if hw.has_cuda else "cpu"
    elif device == "cuda" and not hw.has_cuda:
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    # Apply preset or overrides
    actual_n_envs = n_envs if n_envs is not None else preset.n_envs
    actual_use_subproc = use_subproc if use_subproc is not None else preset.use_subproc
    actual_batch_size = batch_size if batch_size is not None else preset.batch_size
    actual_n_steps = n_steps if n_steps is not None else preset.n_steps
    actual_net_size = NET_SIZE_PRESETS.get(net_size, 64)  # Get hidden layer size from preset
    
    # Default save path in models directory
    if save_path is None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = str(MODELS_DIR / f"lemonade_{algo.lower()}")
    
    # Ensure parent directory exists for custom paths
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"Training {algo.upper()} on LemonadeBench")
    print("=" * 70)
    print(f"Timesteps: {timesteps:,}")
    print(f"Parallel environments: {actual_n_envs}")
    print(f"Vectorization: {'SubprocVecEnv' if actual_use_subproc else 'DummyVecEnv'}")
    print(f"Batch size: {actual_batch_size}")
    print(f"Rollout steps: {actual_n_steps}")
    print(f"Network size: {net_size} ({actual_net_size}x{actual_net_size})")
    print(f"Learning rate schedule: {lr_schedule}")
    print(f"Device: {device.upper()}")
    print(f"Seed: {seed}")
    print()
    
    # Create vectorized training environment
    def make_train_env(i):
        def _init():
            return LemonadeGymEnv(seed=seed + i)
        return _init
    
    if actual_use_subproc:
        # SubprocVecEnv for true parallelism (better for CPU-bound envs)
        train_env = SubprocVecEnv([make_train_env(i) for i in range(actual_n_envs)])
    else:
        # DummyVecEnv for simple sequential execution (lower overhead)
        train_env = DummyVecEnv([make_train_env(i) for i in range(actual_n_envs)])
    
    # Determine learning rate based on schedule
    base_lr = preset.learning_rate
    learning_rate = linear_schedule(base_lr) if lr_schedule == "linear" else base_lr
    
    # Wrap with VecNormalize for observation and reward normalization
    # This is critical for stable training with continuous observations
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    
    # Create evaluation environment (also wrapped with VecNormalize)
    eval_env = DummyVecEnv([lambda: LemonadeGymEnv(seed=seed + 1000)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize reward during eval
        clip_obs=10.0,
        training=False,  # Don't update stats during eval
    )
    
    # Set up callbacks
    callbacks = []
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./{save_path}_best/",
        log_path=f"./{save_path}_logs/",
        eval_freq=max(eval_freq // actual_n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // actual_n_envs, 1000),
        save_path=f"./{save_path}_checkpoints/",
        name_prefix=algo.lower(),
    )
    callbacks.append(checkpoint_callback)
    
    # Check if tensorboard is available
    try:
        import tensorboard
        tb_log = f"./{save_path}_tensorboard/"
    except ImportError:
        tb_log = None
        if verbose:
            print("Note: tensorboard not installed, logging disabled")
    
    # Create model with appropriate hyperparameters
    if algo.lower() == "ppo":
        model = algo_class(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=actual_n_steps,
            batch_size=actual_batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,  # Start without entropy bonus (often better)
            vf_coef=0.5,
            max_grad_norm=0.5,
            normalize_advantage=True,  # Critical for stable training
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[actual_net_size, actual_net_size],
                    vf=[actual_net_size, actual_net_size],
                ),
                activation_fn=torch.nn.Tanh,  # More stable than ReLU for RL
            ),
            device=device,
            verbose=verbose,
            seed=seed,
            tensorboard_log=tb_log,
        )
    elif algo.lower() == "recurrent_ppo":
        # RecurrentPPO with LSTM for temporal reasoning
        # Better for environments with temporal dependencies (inventory, reputation)
        # Note: RecurrentPPO needs shorter n_steps for sequence collection
        recurrent_n_steps = min(actual_n_steps, 256)
        model = algo_class(
            "MlpLstmPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=recurrent_n_steps,
            batch_size=min(actual_batch_size, recurrent_n_steps * actual_n_envs),
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Small entropy bonus helps exploration with LSTM
            vf_coef=0.5,
            max_grad_norm=0.5,
            normalize_advantage=True,
            policy_kwargs=dict(
                lstm_hidden_size=actual_net_size,
                n_lstm_layers=1,  # Single LSTM layer (often sufficient)
                shared_lstm=False,  # Separate LSTM for actor and critic
                enable_critic_lstm=True,  # Critic also uses LSTM
                net_arch=dict(
                    pi=[actual_net_size],
                    vf=[actual_net_size],
                ),
                activation_fn=torch.nn.Tanh,
            ),
            device=device,
            verbose=verbose,
            seed=seed,
            tensorboard_log=tb_log,
        )
    elif algo.lower() == "sac":
        model = algo_class(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            buffer_size=preset.buffer_size,
            learning_starts=1000,
            batch_size=actual_batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(
                net_arch=[actual_net_size, actual_net_size],
            ),
            device=device,
            verbose=verbose,
            seed=seed,
            tensorboard_log=tb_log,
        )
    elif algo.lower() == "a2c":
        model = algo_class(
            "MlpPolicy",
            train_env,
            learning_rate=7e-4,
            n_steps=5,  # A2C works best with small n_steps
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[actual_net_size, actual_net_size],
                    vf=[actual_net_size, actual_net_size],
                ),
            ),
            device=device,
            verbose=verbose,
            seed=seed,
            tensorboard_log=tb_log,
        )
    elif algo.lower() == "td3":
        model = algo_class(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            buffer_size=preset.buffer_size,
            learning_starts=1000,
            batch_size=actual_batch_size,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=dict(
                net_arch=[actual_net_size, actual_net_size],
            ),
            device=device,
            verbose=verbose,
            seed=seed,
            tensorboard_log=tb_log,
        )
    
    print(f"Starting training with {algo.upper()} on {device.upper()}...")
    print()
    
    # Train the model
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save the final model and VecNormalize stats
    model.save(save_path)
    train_env.save(f"{save_path}_vecnormalize.pkl")
    print()
    print(f"Model saved to: {save_path}.zip")
    print(f"VecNormalize stats saved to: {save_path}_vecnormalize.pkl")
    
    # Final evaluation - sync normalization stats to eval env
    print()
    print("=" * 70)
    print("Final Evaluation (10 episodes)")
    print("=" * 70)
    
    # Sync running stats from train to eval env for consistent normalization
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    
    evaluate_model(model, n_episodes=10, seed=seed + 2000, vec_normalize=train_env)
    
    train_env.close()
    eval_env.close()
    
    return model


def evaluate_model(
    model,
    n_episodes: int = 10,
    seed: int = 0,
    render: bool = False,
    vec_normalize: VecNormalize = None,
):
    """
    Evaluate a trained model.
    
    Args:
        model: Trained SB3 model (or path to saved model)
        n_episodes: Number of evaluation episodes
        seed: Starting seed
        render: Whether to render episodes
        vec_normalize: Optional VecNormalize wrapper for observation normalization
    """
    model_path = None
    is_recurrent = False
    
    # Load model if path is provided
    if isinstance(model, str):
        model_path = Path(model)
        # Try to detect algorithm from filename
        for algo_name, algo_class in ALGORITHMS.items():
            if algo_name in model_path.stem.lower():
                model = algo_class.load(model)
                is_recurrent = algo_name == "recurrent_ppo"
                break
        else:
            # Default to PPO
            model = PPO.load(model)
        
        # Try to load VecNormalize stats if they exist
        vecnorm_path = Path(f"{model_path}_vecnormalize.pkl")
        if vecnorm_path.exists() and vec_normalize is None:
            print(f"Loading VecNormalize stats from: {vecnorm_path}")
            # Create a dummy env to load stats into
            dummy_env = DummyVecEnv([lambda: LemonadeGymEnv(seed=seed)])
            vec_normalize = VecNormalize.load(str(vecnorm_path), dummy_env)
            vec_normalize.training = False
            vec_normalize.norm_reward = False
    else:
        # Check if the model is recurrent by checking its class
        is_recurrent = HAS_RECURRENT_PPO and isinstance(model, RecurrentPPO)
    
    profits = []
    cups_sold = []
    
    for i in range(n_episodes):
        if vec_normalize is not None:
            # Use normalized environment
            env = DummyVecEnv([lambda i=i: LemonadeGymEnv(
                seed=seed + i,
                render_mode="human" if render else None,
            )])
            # Create a new VecNormalize with same stats
            eval_env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=False,
                training=False,
            )
            eval_env.obs_rms = vec_normalize.obs_rms
            eval_env.ret_rms = vec_normalize.ret_rms
            
            obs = eval_env.reset()
            
            total_reward = 0
            episode_cups = 0
            
            # Initialize LSTM states for recurrent models
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            
            while True:
                if is_recurrent:
                    # RecurrentPPO needs LSTM states
                    action, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True,
                    )
                    episode_starts = np.zeros((1,), dtype=bool)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, done, info = eval_env.step(action)
                total_reward += reward[0]
                episode_cups += info[0].get("cups_sold", 0)
                
                if done[0]:
                    break
            
            profits.append(info[0]["total_profit"])
            cups_sold.append(episode_cups)
            eval_env.close()
        else:
            # Use raw environment (no normalization)
            env = LemonadeGymEnv(
                seed=seed + i,
                render_mode="human" if render else None,
            )
            obs, info = env.reset()
            
            total_reward = 0
            episode_cups = 0
            
            # Initialize LSTM states for recurrent models
            lstm_states = None
            episode_start = True
            
            while True:
                if is_recurrent:
                    # RecurrentPPO needs LSTM states
                    action, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=np.array([episode_start]),
                        deterministic=True,
                    )
                    episode_start = False
                else:
                    action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                episode_cups += info.get("cups_sold", 0)
                
                if terminated or truncated:
                    break
            
            profits.append(info["total_profit"])
            cups_sold.append(episode_cups)
            env.close()
    
    # Report results
    avg_profit = sum(profits) / len(profits)
    std_profit = (sum((p - avg_profit) ** 2 for p in profits) / len(profits)) ** 0.5
    avg_cups = sum(cups_sold) / len(cups_sold)
    
    print(f"Episodes: {n_episodes}")
    print(f"Average Profit: ${avg_profit / 100:.2f} (+/- ${std_profit / 100:.2f})")
    print(f"Min Profit: ${min(profits) / 100:.2f}")
    print(f"Max Profit: ${max(profits) / 100:.2f}")
    print(f"Average Cups Sold: {avg_cups:.1f}")
    
    return {
        "mean_profit": avg_profit,
        "std_profit": std_profit,
        "min_profit": min(profits),
        "max_profit": max(profits),
        "mean_cups": avg_cups,
        "profits": profits,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train an RL agent on LemonadeBench with automatic hardware optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    algo_help = "RL algorithm to use (default: ppo)"
    if HAS_RECURRENT_PPO:
        algo_help += ". recurrent_ppo uses LSTM for temporal reasoning"
    else:
        algo_help += ". Install sb3-contrib for recurrent_ppo support"
    
    # Algorithm selection
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=list(ALGORITHMS.keys()),
        help=algo_help,
    )
    
    # Training parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel environments (default: auto-detect based on CPU)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (default: auto based on hardware)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Rollout steps per environment (default: auto based on hardware)",
    )
    
    # Performance options
    parser.add_argument(
        "--high-perf",
        action="store_true",
        help="Enable high-performance mode (maximizes hardware utilization)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for training (default: auto-detect)",
    )
    parser.add_argument(
        "--net-size",
        type=str,
        default="small",
        choices=list(NET_SIZE_PRESETS.keys()),
        help="Network size preset: tiny(32), small(64), medium(128), large(256), xlarge(512). "
             "Small is recommended for LemonadeBench's 27-dim observation space. (default: small)",
    )
    parser.add_argument(
        "--use-subproc",
        action="store_true",
        default=None,
        help="Use SubprocVecEnv for true parallelism (default: auto-detect)",
    )
    parser.add_argument(
        "--no-subproc",
        action="store_true",
        help="Force DummyVecEnv (disable SubprocVecEnv)",
    )
    
    # Other training options
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save trained model (default: lemonade_<algo>)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluation frequency in timesteps (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=["constant", "linear"],
        help="Learning rate schedule (default: constant)",
    )
    
    # Evaluation options
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate a trained model instead of training",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model for evaluation (required with --evaluate)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes during evaluation",
    )
    
    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        type=int,
        default=1,
        help="Verbosity level (default: 1)",
    )
    
    # Hardware info only
    parser.add_argument(
        "--show-hardware",
        action="store_true",
        help="Show detected hardware info and exit",
    )
    
    args = parser.parse_args()
    
    # Handle --show-hardware
    if args.show_hardware:
        hw = detect_hardware()
        preset = get_performance_preset(hw, high_perf=False)
        print_hardware_info(hw, preset)
        print("High-Performance Preset:")
        hp_preset = get_performance_preset(hw, high_perf=True)
        print(f"  Parallel Environments: {hp_preset.n_envs}")
        print(f"  Batch Size: {hp_preset.batch_size}")
        print(f"  Rollout Steps: {hp_preset.n_steps}")
        print(f"  Buffer Size: {hp_preset.buffer_size:,}")
        print(f"  Network Size: {hp_preset.net_arch_size}")
        sys.exit(0)
    
    # Handle use_subproc flags
    use_subproc = args.use_subproc
    if args.no_subproc:
        use_subproc = False
    
    if args.evaluate:
        if args.model_path is None:
            print("Error: --model-path is required for evaluation")
            sys.exit(1)
        
        print("=" * 70)
        print(f"Evaluating model: {args.model_path}")
        print("=" * 70)
        
        evaluate_model(
            args.model_path,
            n_episodes=args.n_episodes,
            seed=args.seed,
            render=args.render,
        )
    else:
        train(
            algo=args.algo,
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            save_path=args.save_path,
            eval_freq=args.eval_freq,
            seed=args.seed,
            verbose=args.verbose,
            use_subproc=use_subproc,
            lr_schedule=args.lr_schedule,
            device=args.device,
            high_perf=args.high_perf,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            net_size=args.net_size,
        )


if __name__ == "__main__":
    main()
