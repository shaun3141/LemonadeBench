# LemonadeBench: Sponsorship & Collaboration Proposal

## The Project

**LemonadeBench** is an open-source benchmark for evaluating AI agent decision-making in multi-day business simulations. Unlike existing benchmarks that test single-session tasks (fix a bug, navigate a webpage), LemonadeBench evaluates whether AI agents can sustain coherent strategies over time—where early decisions compound into later constraints through inventory, cash flow, and reputation effects.

Agents operate a lemonade stand over a 14-day season, making daily decisions about pricing, inventory (with perishable goods), location selection, weather adaptation, and capital allocation. The domain is deliberately intuitive: every action—"buy more lemons," "raise the price"—can be understood by anyone, enabling transparent evaluation of AI reasoning.

## Research Contributions

1. **First Goal-Framing Study for LLM Economic Behavior** — Systematic analysis of how motivational prompts (aggressive, conservative, competitive) modulate agent risk tolerance and strategy, with implications for AI alignment

2. **Architecture & Scaffolding Comparisons** — Evaluating whether planning/reflection loops and cognitive tools (calculators, code interpreters) improve long-horizon reasoning

3. **Large-Scale Empirical Study** — 960+ episodes across 20 frontier models (GPT-4o, Claude Sonnet 4, Gemini 2.5 Pro, Llama 3.1 405B, DeepSeek R1, and more)

4. **Open Infrastructure** — Full environment, evaluation harness, interactive web leaderboard, and all data released publicly

## Why This Matters

As LLM agents are deployed in inventory management, pricing optimization, and trading systems, we need rigorous evaluation of their behavior under extended decision horizons. LemonadeBench provides an accessible, reproducible testbed for this critical capability—before agents are deployed in high-stakes economic contexts.

## Sponsorship Request: Up to $1,500 in API Credits

Running comprehensive experiments across frontier models is expensive. Each episode requires multiple LLM calls over 14 simulated days, and rigorous evaluation demands multiple seeds per condition for statistical validity.

**Current experimental scope:**
- 20+ models (GPT-4o, Claude Sonnet 4, Gemini 2.5 Pro, o1, DeepSeek R1, Llama 405B, etc.)
- 6 goal-framing conditions × 4 architectures × 4 scaffolding variants
- 5 seeds per configuration for reproducibility
- **960+ total episodes** — with plans to expand coverage

Premium models like o1, Claude Sonnet 4, and Gemini 2.5 Pro cost $3–75 per million tokens. A single full experimental run can cost $500–1,000+ in API fees.

**Your sponsorship of API credits would directly enable:**
- Broader model coverage (especially expensive reasoning models)
- More seeds for stronger statistical claims
- Additional ablation studies requested by reviewers

**All sponsors will be acknowledged in the published paper and GitHub repository.**

## Collaboration Opportunity

I'm actively seeking collaborators to strengthen the research:

- **Co-authorship** — Contribute to experimental design, analysis, or writing
- **API Credit Grants** — Direct credits or research access programs from model providers
- **Model Access** — Run evaluations on proprietary or unreleased models
- **Domain Expertise** — Behavioral economics, decision theory, or agent architecture insights

If you're interested in advancing our understanding of LLM economic decision-making, I'd welcome the opportunity to collaborate.

---

**Contact:** Shaun Van Weelden — [shaun.t.vanweelden@gmail.com](mailto:shaun.t.vanweelden@gmail.com)  
**GitHub:** [github.com/Shaun3141/LemonadeBench](https://github.com/Shaun3141/LemonadeBench)  
**Live Leaderboard:** [lemonadebench.com](https://lemonadebench.com)

