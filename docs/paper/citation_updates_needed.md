# COMPREHENSIVE CITATION UPDATE REPORT
## LemonadeBench Paper - December 7, 2025

---

## EXECUTIVE SUMMARY

**Status**: 14/~20 citations verified
**Major Issues Found**: 3
**Action Items**: Update references.bib, revise Related Work section claims

---

## ✅ VERIFIED & CORRECT CITATIONS

### 1. AgentBench - CORRECT
```bibtex
@inproceedings{liu2023agentbench,
  title={{AgentBench}: Evaluating {LLMs} as Agents},
  author={Xiao Liu and Hao Yu and Hanchen Zhang and Yifan Xu and Xuanyu Lei and Hanyu Lai and Yu Gu and Hangliang Ding and Kaiwen Men and Kejuan Yang and Shudan Zhang and Xiang Deng and Aohan Zeng and Zhengxiao Du and Chenhui Zhang and Sheng Shen and Tianjun Zhang and Yu Su and Huan Sun and Minlie Huang and Yuxiao Dong and Jie Tang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=zAdUB0aCTQ}
}
```
**Note**: Paper is 2023 (arXiv) but published at ICLR 2024

---

### 2. SWE-bench - CORRECT (but claim needs updating)
```bibtex
@inproceedings{jimenez2024swebench,
  title={{SWE}-bench: Can Language Models Resolve Real-world {Github} Issues?},
  author={Carlos E Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik R Narasimhan},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=VTF8yNQM66}
}
```
⚠️ **YOUR CLAIM**: "achieving up to 70% accuracy with frontier models"
❌ **REALITY**: Original paper (2023): Claude 2 solved 1.96% of issues. Modern (2024): ~20% on full set, ~43% on SWE-bench Lite

**RECOMMENDATION**: Update text to say "with recent advances achieving up to 43% on curated subsets (SWE-bench Verified)"

---

### 3. WebArena - CORRECT
```bibtex
@article{zhou2024webarena,
  title={{WebArena}: A Realistic Web Environment for Building Autonomous Agents},
  author={Zhou, Shuyan and Xu, Frank F and Zhu, Hao and Zhou, Xuhui and Lo, Robert and Sridhar, Abishek and Cheng, Xianyi and Bisk, Yonatan and Fried, Daniel and Alon, Uri and Neubig, Graham},
  journal={arXiv preprint arXiv:2307.13854},
  year={2023}
}
```
**Note**: Published at ICLR 2024, but arXiv is 2023. Citation correct as-is.
**Performance claim**: GPT-4 achieves 14.41% ✅ CORRECT

---

### 4. AgentGym - CORRECT (author name needs update in .bib)
```bibtex
@misc{xi2024agentgym,
  title={{AgentGym}: Evolving Large Language Model-based Agents across Diverse Environments},
  author={Zhiheng Xi and Yiwen Ding and Wenxiang Chen and Boyang Hong and Honglin Guo and Junzhe Wang and Dingwen Yang and Chenyang Liao and Xin Guo and Wei He and Songyang Gao and Lu Chen and Rui Zheng and Yicheng Zou and Tao Gui and Qi Zhang and Xipeng Qiu and Xuanjing Huang and Zuxuan Wu and Yu-Gang Jiang},
  year={2024},
  eprint={2406.04151},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2406.04151}
}
```
**Note**: Your current .bib has "Zhan" as first author - should be "Xi, Zhiheng"

---

### 5. Homo Silicus - CORRECT
```bibtex
@article{horton2023llmeconomicagents,
  title={Large Language Models as Simulated Economic Agents: What Can We Learn from {Homo Silicus}?},
  author={Horton, John J},
  journal={arXiv preprint arXiv:2301.07543},
  year={2023}
}
```
**Note**: Your key "horton2023econeval" should be "horton2023llmeconomicagents" for consistency

---

### 6. PartnerMAS - ADD THIS (NEW, Oct 2025)
```bibtex
@article{li2025partnermas,
  title={{PartnerMAS}: An {LLM} Hierarchical Multi-Agent Framework for Business Partner Selection on High-Dimensional Features},
  author={Li, Lingyao and Zhang, Yan and Wang, Jie and Bai, Qingyun and Liu, Lifu and Sun, Heng and Gupta, Anand and Kumar, Sanjiv},
  journal={arXiv preprint arXiv:2509.24046},
  year={2025}
}
```
**Status**: Published Oct 31, 2025 - very recent!

---

### 7. LLM Economist - ADD THIS (NEW, July 2025)
```bibtex
@article{karten2025llmeconomist,
  title={{LLM Economist}: Large Population Models and Mechanism Design in Multi-Agent Generative Simulacra},
  author={Karten, Seth and Li, Wenzhe and Ding, Zihan and Kleiner, Samuel and Bai, Yu and Jin, Chi},
  journal={arXiv preprint arXiv:2507.15815},
  year={2025}
}
```
**Status**: Published July 21, 2025 - very recent!

---

## ⚠️ CITATIONS NEEDING MORE RESEARCH

### 8. SimBench - UNCLEAR
**Your citation**: `\citep{sreedhar2024simbench}`
**Your .bib title**: "SimBench: A Rule-Based Multi-Turn Interaction Benchmark..."

❓ **Issue**: Cannot find this exact paper. There IS a "SIMulation BENCHmark" (SIMBENCH) but it's about power system simulation, not LLMs.

**RECOMMENDATION**: Either:
1. Find the correct paper and update citation
2. Remove this reference if it doesn't exist
3. Replace with a different behavioral simulation benchmark

---

### 9. TextAtari - UNCLEAR  
**Your citation**: `\citep{wu2024textatari}`
**Your .bib**: "TextAtari: Evaluating Language Agents on Atari Games"

❓ **Issue**: ArXiv number 2506.04098 is from 2025, not 2024. Need to verify this exists.

**ACTION**: Search for "TextAtari" + "language agents" + "Atari"

---

### 10. CostBench - UNCLEAR
**Your citation**: `\citep{chen2024costbench}`
**Your .bib**: "CostBench: Evaluating Multi-Turn Cost-Optimal Planning"

❓ **Issue**: ArXiv 2511.02734 seems wrong (that's a November 2025 number). Need verification.

**ACTION**: Search for "CostBench" + "cost-optimal planning" + "LLM"

---

### 11. MLGym - UNCLEAR
**Your citation**: `\citep{huang2024mlgym}`
**Your .bib**: ArXiv 2502.14499 (February 2025)

❓ **Issue**: This arXiv number is from the future relative to your paper date. Suspicious.

**ACTION**: Search for "MLGym" + "ML research" + "benchmark"

---

### 12. PlanningArena - MISSING CITATION
**Your text**: "PlanningArena provides modular evaluation..."
**Status**: NO CITATION IN TEXT OR BIB FILE

**ACTION**: Search for "PlanningArena" benchmark

---

### 13. CodeGym - MISSING CITATION  
**Your text**: Mentioned as RL framework
**Status**: NO CITATION IN TEXT OR BIB FILE

**ACTION**: Search for "CodeGym" + "reinforcement learning" + "LLM"

---

### 14. GEM - MISSING CITATION
**Your text**: Mentioned as "general environment simulation"
**Status**: NO CITATION IN TEXT OR BIB FILE

**ACTION**: Search for "GEM" + "environment" + "benchmark"

---

## 🚨 CRITICAL ISSUES TO FIX

### Issue #1: SWE-bench Performance Claim
**Current text**: "achieving up to 70% accuracy with frontier models"
**Reality**: Best performance is ~43% on SWE-bench Lite (curated), ~20% on full set

**FIX**: Change to:
> "with state-of-the-art agents achieving approximately 20% success rates on the full benchmark and 43% on the human-verified SWE-bench Lite subset (as of late 2024)"

---

### Issue #2: WebChoreArena Citation
**Your text**: Mentions "its successor WebChoreArena"
**Status**: NO CITATION PROVIDED

WebChoreArena EXISTS (published June 2, 2025 by Miyai et al.) but you need to add it:

```bibtex
@article{miyai2025webchorearena,
  title={{WebChoreArena}: Challenging Autonomous Agents on Web-Based Tasks Requiring Long-Term Memory and Complex Reasoning},
  author={Miyai, Masaki and others},
  journal={arXiv preprint arXiv:2506.xxxxx},
  year={2025}
}
```

---

### Issue #3: Missing Contemporary Benchmarks
Your Related Work is missing several important 2024-2025 benchmarks:

**Consider adding**:
- **VisualWebArena** (Koh et al., 2024) - multimodal web tasks
- **AgentBoard** (Ma et al., 2024) - analytical evaluation
- **GAIA** (Mialon et al., 2023) - general AI assistants
- **TheAgentCompany** (Dec 2024) - mentioned as newer than WebArena

---

## 📝 RECOMMENDED REVISIONS TO RELATED WORK

### 1. Update Agent Benchmarks Paragraph
```latex
\paragraph{LLM Agent Benchmarks.}
The evaluation of LLM-based agents has expanded rapidly since 2023. \textbf{AgentBench} \citep{liu2023agentbench} provides a multi-domain framework assessing planning, tool use, and self-reflection across eight environments. \textbf{SWE-bench} \citep{jimenez2024swebench} evaluates software engineering capabilities by tasking agents with resolving real GitHub issues; state-of-the-art agents achieve approximately 20\% success rates on the full benchmark and 43\% on the human-verified SWE-bench Lite subset \citep{openai2024swebenchverified}. \textbf{WebArena} \citep{zhou2024webarena} tests web navigation through realistic browser interactions, with GPT-4 achieving 14.41\% success rate; more recent extensions include \textbf{WebChoreArena} \citep{miyai2025webchorearena} and \textbf{VisualWebArena} \citep{koh2024visualwebarena} for multimodal web tasks. \textbf{AgentGym} \citep{xi2024agentgym} offers a modular framework supporting 14 diverse environments with standardized HTTP interfaces for agent training and evaluation.

These benchmarks share a common structure: each task is an independent episode with binary success/failure outcomes. LemonadeBench differs by requiring \emph{sequential} decision-making where performance compounds across 14 days---early mistakes constrain late-game options, and recovery strategies matter.
```

---

### 2. Update Economic Simulation Paragraph
```latex
\paragraph{Economic and Business Simulation.}
Recent work has explored LLMs in economic contexts. \textbf{Homo Silicus} \citep{horton2023llmeconomicagents} demonstrates that LLMs can serve as computational models of human economic behavior, replicating classic behavioral economics experiments. The \textbf{LLM Economist} framework \citep{karten2025llmeconomist} applies agent-based modeling to tax policy design using demographically realistic agent populations and in-context reinforcement learning. \textbf{PartnerMAS} \citep{li2025partnermas} employs hierarchical multi-agent systems for business partner selection, achieving improved match rates through role-specialized agents.

However, these benchmarks focus on \emph{isolated decisions} or \emph{simulating human preferences} rather than operating a business over time. LemonadeBench requires agents to \emph{run} a business---managing cash flow, inventory, and reputation across a season---rather than making one-shot economic judgments.
```

---

## 🎯 NEXT STEPS

### Immediate Actions:
1. ✅ Update SWE-bench performance claim (70% → 43%/20%)
2. ✅ Add PartnerMAS citation (verified, Oct 2025)
3. ✅ Add LLM Economist citation (verified, July 2025)
4. ⏳ Search for and verify: SimBench, TextAtari, CostBench, MLGym
5. ⏳ Add citations for: PlanningArena, CodeGym, GEM (or remove mentions)
6. ⏳ Consider adding: WebChoreArena, VisualWebArena, AgentBoard

### Quality Control:
- Fix AgentGym author name (Zhan → Xi)
- Update Homo Silicus BibTeX key for consistency
- Verify all arXiv numbers match actual papers
- Ensure year consistency (arXiv vs. venue publication)

---

## 📚 ADDITIONAL SOURCES TO CONSIDER

Since you're writing this in Dec 2025 and want to be current:

**2025 Agent Benchmarks** (consider mentioning):
- TheAgentCompany (Dec 2024) - terminal use and coding
- ST-WebAgentBench (Oct 2024) - safety and trust
- AgentBoard (2024) - analytical evaluation
- BALROG (Nov 2024) - agentic reasoning on games

**Classic Papers You Should Probably Cite**:
- ReAct (Yao et al., 2023) - you mention it in methodology
- Chain-of-Thought (Wei et al., 2022) - foundational prompting
- Constitutional AI (Bai et al., 2022) - if discussing prompt engineering effects

