# Cooperative Merge Scenario (Phase 1 Baseline)

The **Cooperative Merge** is the foundational scenario that established the core research question in Echo: **Can an autonomous vehicle learn when and whom to ask for hidden driver intent?**

This page summarizes the Phase 1 exploratory scenario. For modern multi-lane highway experiments, see the [Experiments Hub](experiments.html).

---

## 1. The Core Idea & Information Gap

On a two-lane highway, the ego car is forced to merge into adjacent traffic because a stalled vehicle blocks its lane ahead.

The vehicles in the target lane have **hidden behavioral types**:
- **Yielding Driver (+1):** Brakes to create a gap and let the ego merge safely.
- **Aggressive Driver (-1):** Accelerates to close the gap; attempting to merge ahead of it causes a collision.

Before the merge, both drivers look identical kinematically. Watching speeds alone yields an information gap with **51.4% prediction accuracy (pure coin toss)**. Only by sending a communication query can the ego discover the driver's true intent before initiating the merge.

---

## 2. Information Value Benchmark

To prove that communication is worth paying for, hand-written heuristic policies were evaluated across 300 randomized seeds:

| Merge Strategy | Description | Mean Score | Crash Rate |
| :--- | :--- | :---: | :---: |
| **Gamble (Blind)** | Merges immediately without asking | 5.17 | 52% |
| **Never Merge (Timid)** | Stays in lane until stalled car | 4.64 | 0% |
| **Always Ask Rear Car** | Asks the merging neighbor every step | **7.35** | **0%** |

*Takeaway:* Communicating intent completely eliminated merge collisions (52% → 0%) and produced the highest driving reward, proving the value of intent queries.

---

## 3. The Communication Cost Dilemma

In real-world networks, communication isn't free. With a cost penalty of $c = 0.05$ per message:
- **Asking too much:** Wastes rewards asking after the merge is already completed.
- **Asking too little:** Blindly accelerates into aggressive drivers, causing catastrophic crash penalties (-2.0).

This dilemma motivated the development of **Request DQN** and **Value-of-Information (VoI)** policies to selectively query vehicles only when the answer changes the driving decision.

---

## 4. Evolution to Multi-Lane Highway (Phase 2)

While the 2-lane merge validated the information gap concept, real highway driving introduces higher speeds and multi-vehicle traffic:
- **4-lane highways** with surrounding traffic clusters.
- **8 fixed spatial slots** (Left-Front, Beside, Rear, etc.).
- **Dynamic physical safety guards** preventing collisions at 30 m/s.
- **Matched-Intent challenge scenarios** (Experiments 5–7).
