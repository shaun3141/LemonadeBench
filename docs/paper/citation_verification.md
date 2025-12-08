# Citation Verification Report for LemonadeBench Paper
## Related Work Section (Section 2)

Generated: December 7, 2025

---

## VERIFIED CITATIONS

### 1. AgentBench ✅ CORRECT
**Your citation**: `\citep{liu2023agentbench}`
**Verified information**:
- **Authors**: Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, Jie Tang
- **Year**: 2023
- **Venue**: ICLR 2024 (accepted as oral presentation)
- **arXiv**: 2308.03688
- **Claim verification**: "8 distinct environments" ✅ CORRECT

**Correct BibTeX**:
```bibtex
@inproceedings{liu2023agentbench,
  title={{AgentBench}: Evaluating {LLMs} as Agents},
  author={Xiao Liu and Hao Yu and Hanchen Zhang and Yifan Xu and Xuanyu Lei and Hanyu Lai and Yu Gu and Hangliang Ding and Kaiwen Men and Kejuan Yang and Shudan Zhang and Xiang Deng and Aohan Zeng and Zhengxiao Du and Chenhui Zhang and Sheng Shen and Tianjun Zhang and Yu Su and Huan Sun and Minlie Huang and Yuxiao Dong and Jie Tang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=zAdUB0aCTQ}
}
```

---

### 2. SWE-bench ⚠️ NEEDS MINOR UPDATE
**Your citation**: `\citep{jimenez2024swebench}`
**Verified information**:
- **Authors**: Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik R. Narasimhan
- **Year**: Published 2024 at ICLR 2024
- **arXiv**: 2310.06770
- **Claim "up to 70% accuracy"**: ⚠️ **NEEDS VERIFICATION** - Original paper reports ~2% for Claude 2 (2023), but leaderboard shows modern agents at 20-43% on different splits

**Recommendation**: Update the percentage claim or clarify which split/model you're referencing. As of Aug 2024, top agents score ~20% on full SWE-bench, ~43% on SWE-bench Lite

**Correct BibTeX**:
```bibtex
@inproceedings{jimenez2024swebench,
  title={{SWE}-bench: Can Language Models Resolve Real-world {Github} Issues?},
  author={Carlos E Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik R Narasimhan},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=VTF8yNQM66}
}
```

---

### 3. WebArena ⚠️ YEAR DISCREPANCY  
**Your citation**: `\citep{zhou2024webarena}`
**Verified information**:
- **Authors**: Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, Graham Neubig (+ others)
- **arXiv**: 2307.13854 (submitted July 2023)
- **Venue**: ICLR 2024 (though arXiv is 2023)
- **Claim "GPT-4 ~15% success rate"**: ✅ CORRECT (14.41% reported)
- **Note**: You mention "WebChoreArena" as successor - this exists but is very recent (June 2025)

**Correct BibTeX** (note venue year):
```bibtex
@article{zhou2024webarena,
  title={{WebArena}: A Realistic Web Environment for Building Autonomous Agents},
  author={Zhou, Shuyan and Xu, Frank F and Zhu, Hao and Zhou, Xuhui and Lo, Robert and Sridhar, Abishek and Cheng, Xianyi and Bisk, Yonatan and Fried, Daniel and Alon, Uri and Neubig, Graham},
  journal={ICLR},
  year={2024}
}
```

---

### 4. AgentGym ❌ MISSING CITATION
**Your text**: "AgentGym \citep{zhan2024agentgym}"
**Status**: **CITATION NEEDED** - Unable to verify primary paper details

**Action required**: Please provide the full citation details or verify if this is the correct reference

---

## CITATIONS REQUIRING VERIFICATION

### Economic and Business Simulation Section

#### 5. Homo Silicus ❌ CITATION INCOMPLETE
**Your citation**: `\citep{horton2023econeval}`  
**Status**: **NEEDS CORRECTION**

The paper appears to be:
- **Correct title**: "Large Language Models as Simulated Economic Agents: What Can We Learn from Homo Silicus?"
- **Author**: John J. Horton
- **Year**: 2023
- **ArXiv**: Likely exists but key not found as "horton2023econeval"

**Action**: Update BibTeX key and verify correct paper reference

---

#### 6. SimBench ❌ CITATION INCOMPLETE
**Your citation**: `\citep{sreedhar2024simbench}`
**Status**: **CANNOT VERIFY** - No results found

**Action**: Please provide full citation or verify this reference exists

---

#### 7. PartnerMAS ❌ MISSING CITATION
**Your text**: "PartnerMAS uses multi-agent hierarchies..."
**Status**: **NO CITATION PROVIDED**

**Action**: Add complete citation with \citep{}

---

#### 8. LLM Economist ❌ MISSING CITATION
**Your text**: "The LLM Economist framework..."
**Status**: **NO CITATION PROVIDED**

**Action**: Add complete citation with \citep{}

---

### Long-Horizon Planning Section

#### 9. TextAtari ❌ CITATION INCOMPLETE
**Your citation**: `\citep{wu2024textatari}`
**Status**: **CANNOT VERIFY**

**Action**: Provide full citation details

---

#### 10. CostBench ❌ CITATION INCOMPLETE
**Your citation**: `\citep{chen2024costbench}`
**Status**: **CANNOT VERIFY**

**Action**: Provide full citation details

---

#### 11. PlanningArena ❌ MISSING CITATION
**Your text**: "PlanningArena provides..."
**Status**: **NO CITATION PROVIDED**

**Action**: Add complete citation

---

### Reinforcement Learning Environments Section

#### 12. MLGym ❌ CITATION INCOMPLETE  
**Your citation**: `\citep{huang2024mlgym}`
**Status**: **CANNOT VERIFY**

---

#### 13. CodeGym ❌ MISSING CITATION
**Status**: **NO CITATION PROVIDED**

---

#### 14. GEM ❌ MISSING CITATION
**Status**: **NO CITATION PROVIDED**

---

## RECOMMENDATIONS

### IMMEDIATE ACTIONS NEEDED:
1. **Fix SWE-bench percentage claim** - "up to 70%" appears incorrect for the published paper
2. **Add missing citations** for: PartnerMAS, LLM Economist, PlanningArena, CodeGym, GEM
3. **Verify and complete** incomplete citations (AgentGym, SimBench, Homo Silicus, TextAtari, CostBench, MLGym)
4. **Check references.bib file** to ensure all citations have complete BibTeX entries

### SUGGESTED ADDITIONS:
1. Add a new paragraph on **prompt engineering studies** since you're studying goal-framing effects
2. Consider citing **ReAct** (Yao et al., 2023) since you mention it in methodology
3. Consider citing **Chain-of-Thought** (Wei et al., 2022) for reasoning prompts
4. Add **behavioral economics classics** like Kahneman & Tversky for theoretical grounding

---

## NEXT STEPS
1. Provide the references.bib file so I can verify all BibTeX entries are complete
2. Clarify which papers you want to keep vs. remove if citations can't be found
3. I can help search for the missing/incomplete citations if you provide more context

