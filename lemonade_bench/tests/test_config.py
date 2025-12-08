# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""Tests for config deduplication and experimental matrix generation."""

import pytest
from lemonade_bench.harness.config import HarnessConfig, ExperimentConfig, ModelConfig


class TestConfigDeduplication:
    """Test that overlapping experimental configs are properly deduplicated."""

    def test_no_duplicates_in_simple_config(self):
        """Simple config with no overlaps should have equal total and unique counts."""
        config = HarnessConfig(
            name="Test",
            models=[
                ModelConfig(provider="test", name="model1", seeds=[1, 2, 3]),
                ModelConfig(provider="test", name="model2", seeds=[1, 2, 3]),
            ]
        )
        assert config.get_total_runs() == 6
        assert config.get_unique_runs() == 6

    def test_detects_duplicate_configs(self):
        """Same model/seed/conditions appearing twice should be deduplicated."""
        config = HarnessConfig(
            name="Test",
            models=[
                ModelConfig(
                    provider="test", name="model1", seeds=[1, 2],
                    architecture="react", goal_framing="baseline"
                ),
                # Exact duplicate - same everything
                ModelConfig(
                    provider="test", name="model1", seeds=[1, 2],
                    architecture="react", goal_framing="baseline"
                ),
            ]
        )
        assert config.get_total_runs() == 4  # 2 models × 2 seeds
        assert config.get_unique_runs() == 2  # Only 2 unique configs

    def test_different_goal_framing_not_duplicate(self):
        """Same model/seed but different goal framing should not be duplicates."""
        config = HarnessConfig(
            name="Test",
            models=[
                ModelConfig(
                    provider="test", name="model1", seeds=[1],
                    goal_framing="baseline"
                ),
                ModelConfig(
                    provider="test", name="model1", seeds=[1],
                    goal_framing="aggressive"
                ),
            ]
        )
        assert config.get_total_runs() == 2
        assert config.get_unique_runs() == 2  # Different goal framing = unique

    def test_different_architecture_not_duplicate(self):
        """Same model/seed but different architecture should not be duplicates."""
        config = HarnessConfig(
            name="Test",
            models=[
                ModelConfig(
                    provider="test", name="model1", seeds=[1],
                    architecture="react"
                ),
                ModelConfig(
                    provider="test", name="model1", seeds=[1],
                    architecture="plan_act"
                ),
            ]
        )
        assert config.get_total_runs() == 2
        assert config.get_unique_runs() == 2  # Different architecture = unique

    def test_different_scaffolding_not_duplicate(self):
        """Same model/seed but different scaffolding should not be duplicates."""
        config = HarnessConfig(
            name="Test",
            models=[
                ModelConfig(
                    provider="test", name="model1", seeds=[1],
                    tools=[]  # none scaffolding
                ),
                ModelConfig(
                    provider="test", name="model1", seeds=[1],
                    tools=["calculator"]  # calculator scaffolding
                ),
            ]
        )
        assert config.get_total_runs() == 2
        assert config.get_unique_runs() == 2  # Different scaffolding = unique

    def test_ablation_overlap_scenario(self):
        """
        Simulates the paper_methodology.yaml overlap:
        - Architecture ablation tests react/plan_act/etc with baseline/none
        - Scaffolding ablation tests none/calculator/etc with baseline/react
        - Overlap at baseline/react/none
        """
        # Simulated architecture ablation: baseline goal, 2 architectures, none scaffold
        arch_models = [
            ModelConfig(
                provider="test", name="model1", seeds=[1, 2],
                goal_framing="baseline", architecture="react", tools=[]
            ),
            ModelConfig(
                provider="test", name="model1", seeds=[1, 2],
                goal_framing="baseline", architecture="plan_act", tools=[]
            ),
        ]
        
        # Simulated scaffolding ablation: baseline goal, react arch, 2 scaffolds
        scaff_models = [
            ModelConfig(
                provider="test", name="model1", seeds=[1, 2],
                goal_framing="baseline", architecture="react", tools=[]  # OVERLAP!
            ),
            ModelConfig(
                provider="test", name="model1", seeds=[1, 2],
                goal_framing="baseline", architecture="react", tools=["calculator"]
            ),
        ]
        
        config = HarnessConfig(
            name="Test Ablation Overlap",
            models=arch_models + scaff_models
        )
        
        # Total: 4 model configs × 2 seeds = 8
        assert config.get_total_runs() == 8
        
        # Unique: (react, none) × 2 seeds + (plan_act, none) × 2 seeds + (react, calc) × 2 seeds
        # = 2 + 2 + 2 = 6 (the overlap config appears in both ablations)
        assert config.get_unique_runs() == 6
        
        # 2 duplicates removed
        assert config.get_total_runs() - config.get_unique_runs() == 2


class TestPaperMethodologyConfig:
    """Test that paper_methodology.yaml has expected deduplication."""

    def test_paper_methodology_deduplication(self):
        """Verify the paper methodology config has 50 duplicates as expected."""
        from lemonade_bench.harness.config import load_config
        from pathlib import Path
        
        config_path = Path(__file__).parent.parent.parent / "examples" / "paper_methodology.yaml"
        if not config_path.exists():
            pytest.skip("paper_methodology.yaml not found")
        
        config = load_config(config_path)
        
        total = config.get_total_runs()
        unique = config.get_unique_runs()
        duplicates = total - unique
        
        assert total == 1000, f"Expected 1000 total runs, got {total}"
        assert unique == 950, f"Expected 950 unique runs, got {unique}"
        assert duplicates == 50, f"Expected 50 duplicates, got {duplicates}"

