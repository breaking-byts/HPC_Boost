# The AI Models Explained (XGBoost, Isolation Forest, OCSVM)

To detect malware using our 24 hardware features, we tested three very different AI algorithms. It is crucial to understand the difference between **Supervised** and **Unsupervised** models, as this was the key to understanding our results.

---

## 1. The Core Difference: Supervised vs. Unsupervised

### Supervised Learning (The "Flashcard" Method)
* **How it works:** You give the AI thousands of examples of Benign programs *and* thousands of examples of Malware, fully labeled. The AI studies both sides and learns the exact differences between them.
* **Pros:** Extremely accurate because it knows exactly what the "enemy" looks like.
* **Cons:** It struggles with "Zero-Day" malware (brand new viruses that look nothing like the training data).

### Unsupervised / Anomaly Detection (The "Bouncer" Method)
* **How it works:** You train the AI **ONLY on Benign (normal) programs**. You don't show it any malware at all during training. 
* **The Logic:** The AI learns exactly what a "normal, healthy computer" looks like. During testing, if a program behaves in a way the AI has never seen before, it throws a red flag and says: *"I don't know what this is, but it's an anomaly!"*
* **Pros:** Excellent for catching Zero-Day malware, because it doesn't need to know what the virus looks like beforehand; it just knows it's not normal.
* **Cons:** Generally lower accuracy than supervised models, and higher false positives (it might flag a heavy video game as "weird").

---

## 2. Model 1: XGBoost (Supervised)
* **Full Name:** Extreme Gradient Boosting
* **What it is:** The heavy-weight champion of modern Machine Learning. 
* **How it works:** It builds a "Decision Tree" (a flowchart of yes/no questions like "Is Cache Miss > 5000?"). But one tree isn't very smart. So, XGBoost builds *hundreds* of trees. The magic is that every time it builds a new tree, it specifically focuses on fixing the mistakes made by the previous trees. It "boosts" its own intelligence step-by-step.
* **Our Results:** This was our best performer. HPC-Boost achieved over **90% accuracy (0.9026 F1)** with XGBoost, beating all the other baselines.

---

## 3. Model 2: Isolation Forest (Unsupervised)
* **What it is:** An algorithm specifically designed to find anomalies.
* **How it works:** Imagine a crowded room of people (normal data) and one person standing alone in the corner (an anomaly). If you draw random lines through the room, you will isolate the person in the corner very quickly, with just a few lines. Isolating someone in the middle of the crowd takes many lines.
* **The Math:** The AI builds random decision trees to chop up the data. Normal benign programs get buried deep in the tree. Malware (which acts weirdly) gets separated out almost immediately at the top of the tree. If it gets isolated fast, the AI flags it as malware.
* **Our Results:** Trained only on benign data. HPC-Boost achieved around **61% accuracy** here.

---

## 4. Model 3: One-Class SVM (Unsupervised)
* **Full Name:** One-Class Support Vector Machine
* **What it is:** A mathematical boundary-drawing algorithm.
* **How it works:** Think of it like a sheepdog corralling sheep. We map all the features of our benign programs onto a giant multidimensional graph. The OCSVM draws a tight mathematical "fence" around all the normal benign points. 
* **The Logic:** When we test a new program, we plot it on the graph. If it lands inside the fence, it's Benign. If it lands outside the fence, it's an anomaly (Malware).
* **Our Results:** Also trained only on benign data. HPC-Boost achieved around **59% accuracy** here.

---

## The Presentation Takeaway for the Professor
*"We used XGBoost to prove that HPC-Boost achieves State-Of-The-Art accuracy (90%+) when we have known malware signatures. We then purposefully handicapped our system by using Unsupervised models (Isolation Forest and OCSVM). We did this to prove that even if we train our system exclusively on normal software, it can still detect malware purely by observing anomalous hardware spikes. This proves HPC-Boost is viable for Zero-Day threat detection."*
