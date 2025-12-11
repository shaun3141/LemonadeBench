# 🍋 LemonadeBench

**A benchmark for evaluating AI agent decision-making through multi-day business simulation.**

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-green.svg)](https://github.com/meta-pytorch/OpenEnv)

<p align="center">
  <a href="#-the-challenge">The Challenge</a> •
  <a href="#-research-paper">Paper</a> •
  <a href="#-leaderboard">Leaderboard</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-game-mechanics">Game Mechanics</a>
</p>

---

## 🎯 The Challenge

**Can LLMs make sustained, strategic business decisions?**

Existing AI benchmarks test single-session task completion: fix a bug, navigate a website, answer questions. But real-world decision-making requires **sustained reasoning over extended time horizons** where early choices constrain future possibilities through resource accumulation, reputation effects, and opportunity costs.

LemonadeBench evaluates AI agents on exactly this capability. Agents operate a lemonade stand business over a 14-day summer season, making daily decisions about:

| Decision Area | What It Tests |
|---------------|---------------|
| **Pricing Strategy** | Balancing profit margins against customer conversion |
| **Inventory Management** | Forecasting demand while avoiding perishable spoilage |
| **Location Selection** | Risk/reward tradeoffs across venues |
| **Capital Allocation** | Investing in upgrades vs. maintaining liquidity |
| **Weather Adaptation** | Adjusting strategy to stochastic conditions |

The lemonade stand domain is deliberately chosen for **interpretability**—every action maps to intuitions humans universally understand: "buy more lemons," "raise the price," "move to a busier location."

**Building on prior work:** Multi-day business simulation is an emerging LLM evaluation paradigm. [Vending-Bench](https://arxiv.org/abs/2502.15840) and [retail management simulators](https://arxiv.org/abs/2509.26331) have established this as a valuable testbed. LemonadeBench extends this direction with richer game mechanics (location selection, perishable inventory, reputation systems) and—critically—systematic studies of *how* LLMs behave: goal-framing effects, architecture comparisons, and cognitive scaffolding analysis.

---

## 📄 Research Paper

> **LemonadeBench: A Dynamic Benchmark for Evaluating AI Agent Decision-Making in Simulated Business Environments**
>
> Shaun Van Weelden

### Key Contributions

1. **Goal-Framing Study** — First systematic study of how motivational prompts (aggressive, conservative, competitive, survival, growth) affect LLM economic behavior—revealing whether prompt engineering can modulate agent risk tolerance, pricing aggression, and strategic planning

2. **Architecture Comparison** — Comprehensive evaluation of four agent loop structures (ReAct, Plan-Act, Act-Reflect, Full) to determine if explicit planning or reflection improves long-horizon business performance

3. **Cognitive Scaffolding Analysis** — Systematic testing of whether calculator tools, math encouragement prompts, or code interpreters improve agent reasoning in economic contexts

4. **Rich Business Simulation** — Multi-day sequential decision-making with compounding effects, perishable inventory, stochastic weather, location-based risk/reward tradeoffs, and delayed reputation feedback—extending prior business simulation benchmarks like [Vending-Bench](https://arxiv.org/abs/2502.15840) with richer game mechanics

### Experimental Design

| Experiment | Models | Conditions | Seeds | Total Episodes |
|------------|--------|------------|-------|----------------|
| Goal Framing | 20 | 6 framings | 5 | 600 |
| Architecture | 4 | 4 architectures | 10 | 160 |
| Scaffolding | 4 | 4 scaffoldings | 10 | 160 |

**Paper:** `docs/paper/` | **Data:** Available in the [Leaderboard](https://lemonadebench.com/leaderboard)

---

## 🏆 Leaderboard

Explore benchmark results across 20 frontier LLMs at **[lemonadebench.com/leaderboard](https://lemonadebench.com/leaderboard)**

The interactive leaderboard features:
- **Results by Goal Framing** — How do aggressive vs. conservative prompts affect profit?
- **Architecture Comparison** — Does planning or reflection help?
- **Scaffolding Analysis** — Do calculators or code interpreters improve reasoning?
- **Detailed Run Inspection** — Day-by-day breakdown of agent decisions and reasoning

### Model Tiers Evaluated

| Tier | Example Models | Price Range |
|------|----------------|-------------|
| **Premium** | Claude Sonnet 4, o1, GPT-4o, Gemini 2.5 Pro | $3–75/M tokens |
| **Balanced** | Claude 3.5 Sonnet, o3-mini, Gemini 2.5 Flash | $1–6/M tokens |
| **Value** | DeepSeek R1, DeepSeek Chat, QwQ-32B | $0.14–2.19/M tokens |
| **Open Source** | Llama 3.3 70B, Llama 3.1 405B, Qwen 2.5 72B | $0.20–2/M tokens |
| **Fast** | Claude 3.5 Haiku, GPT-4o-mini, Gemini Flash 1.5 | $0.075–4/M tokens |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Shaun3141/LemonadeBench.git
cd LemonadeBench

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Run a Simple Agent

```bash
# Run a single episode with the rule-based agent
python examples/simple_agent.py

# Benchmark across 100 episodes
python examples/simple_agent.py --benchmark --episodes 100
```

### Run an LLM Agent

```bash
# Single model test
uv run harness examples/single_model_test.yaml

# Full methodology (requires API keys)
uv run harness examples/paper_methodology.yaml
```

### Use Programmatically

```python
from lemonade_bench import LemonadeAction, LemonadeObservation
from lemonade_bench.server.lemonade_environment import LemonadeEnvironment

# Create environment with reproducible seed
env = LemonadeEnvironment(seed=42)
obs = env.reset()

print(f"Day {obs.day}: {obs.weather}, {obs.temperature}°F")
print(f"Starting cash: ${obs.cash / 100:.2f}")

# Make daily decisions
while not obs.done:
    # Use quantity_to_tier_count to auto-apply bulk discounts
    from lemonade_bench.models import quantity_to_tier_count
    lt, lc = quantity_to_tier_count("lemons", 10)
    st, sc = quantity_to_tier_count("sugar", 3)
    ct, cc = quantity_to_tier_count("cups", 30)
    
    action = LemonadeAction(
        price_per_cup=100,                  # $1.00 per cup
        lemons_tier=lt, lemons_count=lc,    # Restock lemons
        sugar_tier=st, sugar_count=sc,      # Restock sugar
        cups_tier=ct, cups_count=cc,        # Restock cups
        advertising_spend=50,   # $0.50 on ads
    )
    obs = env.step(action)
    print(f"Day {obs.day}: Sold {obs.cups_sold} cups, profit ${obs.daily_profit/100:.2f}")

print(f"Final profit: ${obs.total_profit / 100:.2f}")
```

---

## 🎮 Game Mechanics

### Season Structure

- **Duration:** 14 days (configurable)
- **Goal:** Maximize total profit
- **Starting Cash:** $20.00
- **Weather:** Stochastic, with next-day forecast provided

### Weather System

| Weather | Temperature | Demand Multiplier | Customer Behavior |
|---------|-------------|-------------------|-------------------|
| 🔥 Hot | 90–105°F | 1.8× | Peak demand, ice bonus active |
| ☀️ Sunny | 75–90°F | 1.3× | Above average |
| ☁️ Cloudy | 65–80°F | 0.9× | Slightly below average |
| 🌧️ Rainy | 55–70°F | 0.4× | Low foot traffic |
| ⛈️ Stormy | 50–65°F | 0.1× | Near-zero customers |

### Inventory & Costs

| Item | Cost | Yield | Expiration |
|------|------|-------|------------|
| 🍋 Lemon | $0.25 | 4 cups | 3 days (FIFO) |
| 🍚 Sugar Bag | $0.50 | 10 cups | Never |
| 🥤 Cup | $0.05 | 1 serving | Never |
| 🧊 Ice Bag | $0.25 | 5 cups | Overnight |

### Bulk Discounts (Auto-Applied)

Specify the quantity you want - the system automatically applies the best bulk discount:

| Supply | 10% off threshold | 20% off threshold |
|--------|-------------------|-------------------|
| Lemons | 12+ (Dozen) | 144+ (Crate) |
| Sugar | 5+ (Case) | 20+ (Pallet) |
| Cups | 50+ (Sleeve) | 250+ (Case) |
| Ice | 5+ (Cooler Pack) | 20+ (Delivery) |

**Note:** Lemonade is made **on-demand** as customers arrive—no need to pre-commit to production quantities.

### Demand Model

Customer demand follows a two-stage funnel:

```
Demand = FootTraffic(weather, location, reputation, ads) × Conversion(price, weather, ice)
```

- **Foot Traffic** scales with location choice, weather, and reputation (0.5×–1.5× multiplier)
- **Conversion** decreases as price rises above $0.50 (optimal price point)
- **Ice Bonus:** +20% conversion on hot/sunny days when ice is available

### Reputation System

- Starts at 0.5 (neutral)
- Updates daily: `reputation = 0.8 × previous + 0.2 × today's_satisfaction`
- Satisfaction based on pricing fairness and avoiding stockouts
- Creates delayed feedback—rewards consistent performance over short-term exploitation

### Locations

| Location | Traffic | Price Sensitivity | Weather Exposure | Permit |
|----------|---------|-------------------|------------------|--------|
| 🌳 Park | 1.2× | 0.018 | Full | Free |
| 🏙️ Downtown | 1.0× | 0.012 | 0.7× | $10.00 |
| 🛒 Mall | 0.7× | 0.008 | None | $15.00 |
| 🏊 Pool | 0.9× | Variable* | 1.8× | $2.50 |

*Pool has reduced price sensitivity (0.010) on hot/sunny days.

---

## 🏗️ Architecture

LemonadeBench follows the **OpenEnv** specification for containerized RL environments:

```yaml
# openenv.yaml
spec_version: 1
name: lemonade_bench
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

### Project Structure

```
lemonade_bench/
├── __init__.py                 # Package exports
├── models.py                   # Action, Observation, Config dataclasses
├── client.py                   # HTTP client for remote environments
├── server/
│   ├── app.py                  # FastAPI server
│   └── lemonade_environment.py # Core game logic
├── agents/                     # Agent implementations
│   ├── architectures/          # ReAct, Plan-Act, Act-Reflect, Full
│   └── providers/              # Anthropic, OpenAI, OpenRouter
├── harness/                    # Experiment runner
│   ├── runner.py               # Batch execution
│   └── config.py               # YAML config parsing
└── db.py                       # Supabase integration for results
```

### Running the Server

```bash
# Local development
cd lemonade_bench
uv run server --host 0.0.0.0 --port 8000

# Using Docker
docker build -t lemonade-bench .
docker run -p 8000:8000 lemonade-bench
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/reset` | POST | Start a new game |
| `/api/step` | POST | Submit action, advance day |
| `/health` | GET | Health check |

---

## 🧪 Running Experiments

### Configuration Format

Experiments are defined in YAML:

```yaml
# examples/goal_framing_experiment.yaml
name: goal_framing_study
seeds: [1, 2, 3, 4, 5]

models:
  - id: anthropic/claude-sonnet-4
    provider: anthropic
  - id: openai/gpt-4o
    provider: openai

goal_framings:
  - baseline
  - aggressive
  - conservative
  - competitive
  - survival
  - growth

architecture: react
scaffolding: none
```

### Running

```bash
# Run all combinations
uv run harness examples/goal_framing_experiment.yaml

# Results saved to runs/ directory and uploaded to Supabase
```

---

## 🌐 Web Client

An interactive web interface for human play and result visualization:

```bash
cd web-client
npm install
npm run dev
```

**Features:**
- Human-friendly game interface
- Market intelligence dashboard
- Day-by-day history
- Leaderboard exploration

---

## 📊 Reward Structure

- **Daily reward:** `daily_profit / 100` (profit in dollars)
- **End-of-game bonus:** `total_profit / 1000` (encourages long-term optimization)

A well-performing agent achieves **$30–50+ total profit** over the 14-day season.

---

## 🔬 Reproducibility

All randomness (weather, customer variance) is **pre-generated from the seed** at episode start, ensuring identical conditions across runs with the same seed. This enables:

- Fair comparison across agents
- Ablation studies isolating specific variables
- Exact reproduction of any result

---

## 📚 Citation

```bibtex
@misc{vanweelden2025lemonadebench,
  title={LemonadeBench: A Dynamic Benchmark for Evaluating AI Agent Decision-Making in Simulated Business Environments},
  author={Van Weelden, Shaun},
  year={2025},
  url={https://github.com/Shaun3141/LemonadeBench}
}
```

---

## 📜 License

BSD-3-Clause License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>🍋 Make some lemonade! 🍋</strong>
  <br>
  <a href="https://lemonadebench.com">Website</a> •
  <a href="https://lemonadebench.com/leaderboard">Leaderboard</a> •
  <a href="https://github.com/Shaun3141/LemonadeBench/issues">Issues</a>
</p>
