# 🍋 LemonadeBench

A Lemonade Stand Tycoon simulation environment built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) for reinforcement learning research.

## Overview

LemonadeBench simulates a classic lemonade stand business where an RL agent must learn to:

- **Set optimal prices** based on weather conditions
- **Manage inventory** efficiently (lemons, sugar, cups)
- **Build reputation** through customer satisfaction
- **Maximize profit** over a 14-day summer season

## Quick Start

```python
from lemonade_bench import LemonadeEnv, LemonadeAction

# Connect to a running server
client = LemonadeEnv(base_url="http://localhost:8000")

# Reset and play
result = client.reset()
print(f"Day {result.observation.day}: {result.observation.weather}")

action = LemonadeAction(price_per_cup=75, cups_to_make=30)
result = client.step(action)
print(f"Sold {result.observation.cups_sold} cups!")
```

## Game Mechanics

### Weather System

| Weather | Demand Multiplier |
|---------|-------------------|
| 🔥 Hot | 1.8x |
| ☀️ Sunny | 1.3x |
| ☁️ Cloudy | 0.9x |
| 🌧️ Rainy | 0.4x |
| ⛈️ Stormy | 0.1x |

### Costs

- Lemon: $0.25 (makes 4 cups)
- Sugar bag: $0.50 (makes 10 cups)
- Cup: $0.05
- Ice bag: $0.25 (makes 5 cups)

### Bulk Discounts

Bulk discounts are auto-applied based on quantity:
- 10% off at tier 2 thresholds (dozen lemons, 5 sugar/ice, 50 cups)
- 20% off at tier 3 thresholds (crate lemons, 20 sugar/ice, 250 cups)

## API

See the main [README](../README.md) for full documentation.


