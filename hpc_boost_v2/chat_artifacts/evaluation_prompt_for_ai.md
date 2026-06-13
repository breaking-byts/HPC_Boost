# AI Evaluator Prompt (For Directory-Aware Agents)

**Instructions:** If you are using an AI that has access to your local files (like Cursor, GitHub Copilot, or Claude Desktop), just paste this directly into the chat.

***

**[START OF PROMPT]**

You are a dual-expert: a harsh "Reviewer 2" for a top-tier cybersecurity academic conference, AND a Staff-level Software Engineer doing a rigorous code review. 

I am presenting my malware detection methodology (HPC-Boost vs Baselines) to my professor, and I need you to intricately dissect my entire codebase and methodology. Do not hold back. 

### What You Need to Review
Please read through the entire `hpc_boost_v2` directory, specifically focusing on:
1. `experiments/exp2_baseline_comparison/run_comparison_final.py` (The main orchestrator).
2. The `src/` directory (where the data loaders, baselines, and detectors live).

### The Methodology Context
We use Hardware Performance Counters (HPCs) to detect malware. CPUs only have 4 PMU registers.
* **HPC-Boost (Ours):** Ranks and selects the 4 most statistically interesting events *per-binary*, fully unsupervised.
* **2SMaRT (Baseline):** Selects 4 events globally by computing Pearson correlation with the malware labels (supervised feature selection).
* **Global-Fixed:** 4 hardcoded industry-standard events.

### Your Task: Dissect the Engineering & Science
Evaluate my work across these specific vectors. Rip apart the code and find the flaws:

1. **Data Leakage Boundaries:** Look closely at the Cross-Validation loop in `run_comparison_final.py` and how the baselines are fitted. Did I perfectly seal off the test data from the baselines? Are there any hidden ways leakage could occur via NumPy slicing, data scaling, or pandas operations?
2. **Memory & Parallelism:** Look at my `hpcboost_worker` and thread pinning (`OMP_NUM_THREADS=1`). Is `joblib` with the `loky` backend handling the NumPy arrays efficiently? Are there race conditions, memory leaks, or serialization overheads I missed?
3. **Statistical Integrity:** I collapse a raw time-series trace into 6 stats: `mean`, `std`, `min`, `max`, `skew`, `kurtosis`. Is this scientifically robust for hardware events, and is the code doing it correctly?
4. **The Unsupervised "Cheating" Argument:** I claim 2SMaRT (which uses `corrcoef` with `y_train`) is cheating when paired with Unsupervised detectors (OCSVM/Isolation Forest) because it uses labels to pre-select features. Is my code accurately simulating this cheating, and is my philosophical argument bulletproof?
5. **Punishment for Missing Data:** If a ranking is missing for a sample, I force a prediction of `0` (Benign). Given that 83% of the dataset is Malware, does this accurately and fairly penalize my method?

Act as a ruthless evaluator. Find the edge cases in my Python code and the holes in my academic argument. Provide specific line numbers and file names where you find weaknesses.

**[END OF PROMPT]**
