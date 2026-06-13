# Why 2SMaRT "Cheats" on Unsupervised Learning

When you present to your professor, they will likely ask: *"If 2SMaRT beats HPC-Boost on unsupervised models, why is HPC-Boost better?"* 

To answer this, you have to explain the concept of **Supervised Feature Selection**. Here is the exact breakdown of why 2SMaRT's victory here is fundamentally flawed for real-world zero-day detection.

---

## 1. What is a "Label"?
In our dataset, every single binary comes with a ground-truth answer key called a **Label**. 
* If the binary is normal software, its label is `0` (Benign).
* If the binary is a virus, its label is `1` (Malware).

## 2. The Rule of Unsupervised Learning
The entire point of an **Unsupervised** model (like Isolation Forest or OCSVM) is that it is **blind to the labels**. 
* We only feed it Benign (`0`) data during training so it learns what "normal" looks like. 
* We want it to catch "Zero-Day" malware—brand new viruses that we've never seen before. Because we've never seen them, we don't have labels for them. Unsupervised models are supposed to catch these viruses purely because they act "abnormal."

## 3. How 2SMaRT "Cheats" (The Misuse)
Before we can train the AI, we have to pick 4 hardware events out of the 55 available. This is where 2SMaRT breaks the rules.

* **Step 1:** 2SMaRT looks at the *entire training dataset* and **looks at the labels**.
* **Step 2:** It runs a mathematical formula (Pearson Correlation) to ask: *"Out of all 55 events, which 4 events perfectly spike up when the label is `1`, and stay flat when the label is `0`?"*
* **Step 3:** It selects those 4 perfectly correlated events.
* **Step 4:** It then hands those 4 perfect events to the "Unsupervised" AI.

**The Flaw:** The Unsupervised AI looks incredibly smart because it easily spots the anomalies. But it only spotted them because 2SMaRT used the answer key (the labels) to pre-select the absolute easiest features to look at! 

## 4. Why HPC-Boost is the True Winner
HPC-Boost plays strictly by the rules. 
* When HPC-Boost picks its 4 events, it **never looks at the labels**. 
* It simply looks at the raw data trace of the binary and asks: *"Which of these 55 events are mathematically spiky or volatile right now?"* 

## The Ultimate Argument for your Professor
*"Professor, 2SMaRT achieves a higher F1 score on Unsupervised models, but it does so by using a Supervised event selection method. It uses the ground-truth labels to find the best events. In a real-world, Zero-Day scenario where a new virus hits a system, we do not have a label. 2SMaRT's method of picking events would completely fail because it relies on knowing what the malware looks like beforehand.*

*HPC-Boost is fully label-agnostic. It ranks and selects events dynamically based purely on the real-time statistical behavior of the binary. Therefore, HPC-Boost is the only method that is truly viable for unsupervised, Zero-Day threat detection."*
