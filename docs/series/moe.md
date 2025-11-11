
# The Sparsity Frontier: Core Innovations in Kimi K2 and DeepSeek V3
(NOTE: UNDER CONSTRUCTION)

## 1. Introduction
A core design limitation of standard <span class="idea">dense</span> LLMs is the tight coupling between parameter count ("knowledge" or "long-term memory"), and flop count (the amount of "work" required to generate a token), due to knowledge being diffuse across parameters rather than localized.
When we ask "what is the capital of Kenya?", we effectively pay for the LLM's
proficiency in Medieval history, Korean literature, and quantum mechanics, despite the bounded scope
of our question.

A natural solution to this problem is <span class="idea">sparsity</span>: decouple memory and
compute by selectively invoking a bounded subset of parameters for each layer-token pair,
thereby implicitly localizing knowledge. <span class="term">Mixture of Experts (MoE)</span> layers are a particular
class of solutions in the design space of sparse networks, and are the standard approach used today. 

The core challenge of sparsity in general and MoEs in particular is that the imposition of discrete
structure on systems we train with continuous tools (backprop) creates a fundamental tension, with
implications for learning dynamics, manifold geometry, memory access, inter-GPU communication, and
post-training. Moreover, these problems are tightly coupled; hence progress in sparsity
necessitates expertise in modeling and systems alike. In this two-part series, we'll
dig into these core challenges, review some of the foundational approaches to them, and explore
several novel ideas introduced in the bleeding edge sparse models <span class="term">DeepSeek
V3</span> and <span class="term">Kimi K2</span>.


## 2. Preliminaries
### 2.1 Prerequisites
This is a fairly technical article, and we'll assume familiarity with the fundamentals of MoEs and
distributed training (e.g. 4D parallelism (PP/DP/EP/TP), ZeRO, etc.). We'll draw on some ideas from our previous article on [transformers as information flow graphs](https://github.com/PranavSriram18/genai_foundations/blob/main/docs/series/transformer_internals.md); while
this isn't strictly required pre-reading, we recommend skimming Section 3. 

### 2.2 Scope
While Kimi and DeepSeek introduce a number of innovations across the training stack, we'll primarily
focus on those directly connected to MoEs. In particular, we'll discuss:
* challenges with training large sparse models, and frames for reasoning about them (Section 3)
* elements of Kimi K2 and DeepSeek V3's architectures that utlize existing approaches to sparsity
(Section 4)
* A novel approach to auxiliary-free load balancing introduced in DeepSeek V3 (Section 5)
* Other innovations in K2 and V3, such as dispersion bounding, finer-grained
sparsity, MLA, MuonClip, etc. (Brief overview in Section 6; deeper dives in Part 2).

### 2.3 Notation
The table below standardizes the notation we'll be using throughout. Note that throughout when discussing
"active" or "fixed" experts/params we mean per token.

| Symbol | Meaning                                                               |
| ------ | --------------------------------------------------------------------- |
| $D$    | Embedding dimension (model hidden size per token)                     |
| $T$    | Context length (tokens per sequence)                                  |
| $m$    | Total number of routed experts (excludes shared expert unless stated) |
| $k$    | Active routed experts                                       |
| $k_f$  | Fixed experts (e.g., shared expert)                         |
| $b$    | Expert width (neurons in a single expert’s MLP/FFN block)             |
| $w$    | MoE width ratio ( $w = \frac{b \cdot m}{D}$ )                         |
| DV3    | Abbreviation for Deepseek V3                                          | 
| K2     | Abbreviation for Kimi K2                                              |


---

## 3. The Challenge of Sparsity
In this section, we briefly frame the core challenges involved with sparse models and outline common approaches to mitigating them. 

### 3.1 Communication Complexity
Wide, sparsely activated layers control compute at the cost of more complex communication. Under
<span class="term">expert parallelism</span>, individual experts live on different devices,
necessitating per-token dynamic dispatching to the $k$ active experts. In addition to compute vs.
communication tradeoffs, the heterogeneity of communication costs introduces a new dimension of
complexity: intra-node <span class="term">NVLink/NVSwitch</span> has comparatively high bandwidth
and low latency compared to inter-node <span class="term">RoCE/IB</span>, and large
models require both.

Naive dynamic routing can splice a sequence across many nodes, producing fragmented communication
patterns (sparse access, scattered gathers, small kernels) with poor cache locality. Pipeline
schedules such as [1F1B](TODO - link) or [DualPipe](TODO - link) attempt to overlap these exchanges
with compute, but deep pipelines introduce bubbles unless the overlap is precise. Strategies to
mitigate these issues, such as activation recomputation, lower-precision storage, CPU offloading,
shared expert caching, and router replication each introduce additional complexity and possible
interactions with other aspects of implementation and analysis.

### 3.2 Experts, Manifold Partitioning, and a Fishing Analogy
TODO - this part needs substantial changes
The promise of MoE rests on <span class="term">expert specialization</span>. Specialization fails when <span class="term">dead experts</span> emerge from gradient starvation, when a few experts monopolize traffic, or when routing boundaries jitter so much that experts cannot settle on stable roles. Classic <span class="term">auxiliary load-balancing losses</span> spread utilization but can corrupt token–expert affinities and weaken specialization. Temperature, entropy, and EMA schedules trade exploration against lock-in. Capacity limits force second-best choices when the top expert is full, which can help exploration but also increase variance. A <span class="term">shared expert</span> can backstop quality, yet too much shared capacity reduces pressure to specialize. Curriculum and post-training shift the data manifold, moving Voronoi-like boundaries and destabilizing routes unless the router adapts.

These modeling choices are not free of systems constraints. If we cap a token to ≤ <span class="term">M</span> nodes to control dispersion, routing becomes a constrained assignment problem. Scores must be aggregated by node, nodes are selected under the cap, and experts are filled within those nodes to reach top-k. Stronger balancing stabilizes throughput but risks misrouting. Larger expert pools improve coverage but add dispatcher metadata and increase small-message traffic. <span class="idea">The practical objective is the best expert assignment given a communication and
capacity budget, with gradients that remain faithful to token–expert affinity</span>.


---
## 4. Architecture Specs and Standard Elements

### 4.1 Spec Table
Below is a table detailing core hyperparamerers for DV3 and K2.  

| Dimension                         | Deepseek V3                                                                | Kimi K2                                                           |
| --------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Total Params**                  | **[671B]**                                                         | **[1.0T]**                                                |
| **Active Params**     | **[37B]**                                                          | **[32B]**                                              |
| **Total Experts**                 | **256 routed + 1 shared**                                                  | **384 routed + 1 shared**                                         |
| **Active Experts**        | **8 routed + 1 shared**                        | **8 routed + 1 shared**                                  |
| **Expert width (per-expert MLP)** | **[VERIFY hidden size / FFN dims]**                                        | **[VERIFY hidden size / FFN dims]**                               |
| **Learning algorithm**            | Next-token LM with **MTP**; aux-free balancing on router                   | Next-token LM; training stabilized with **MuonClip / QK-Clip**    |
| **Routing control**               | **Aux-loss-free dynamic bias** + minimal **sequence-level** safeguard      | **Simple top-k** (no grouping, no in-gate balancing)              |
| **Dispersion control**            | **Node-limited (≤ M nodes/token)** — single-pass, effectively hierarchical | None explicit; relies on schedule + memory tactics                |
| **Attention / KV**                | **MLA**                                                                    | **MLA**                                                           |
| **Parallelism / overlap**         | **PP + EP**, little/no TP; **DualPipe-style** compute↔all-to-all overlap   | **PP (virtual) + EP + ZeRO-1**, **interleaved 1F1B**, no DualPipe |
| **Memory / activation tactics**   | Recompute; shared-expert proximity/caching **[VERIFY]**                    | **Selective recompute**, **FP8 activations**, **CPU offload**     |
| **Traffic-shaping policy**        | Keep paths **intra-node** via node-limit; bounded cross-node fan-out       | Hide EP traffic under compute; RoCE-centric scheduling            |

### 4.2 Standard Pieces

Before we dive into novel ideas, let’s briefly establish the common elements that both DeepSeek V3 and Kimi K2 share with standard MoE architectures.

* <span class="term">Experts</span>: feed-forward MLP specialists trained under **top-k** gating with **per-expert capacity** limits. A small <span class="term">shared expert</span> provides general coverage when routed experts miss.

* <span class="term">Routing core</span>: token→expert scoring produces **logits per expert**, followed by **top-k selection** under capacity; typical stabilizers include temperature schedules and light stochasticity; dispatch is batched for EP efficiency.

* <span class="term">Training parallelism</span>: both use **pipeline parallelism**, **expert parallelism**, and **ZeRO-1 data parallelism** with expert sharding. This is the baseline distributed recipe for large MoEs.

* <span class="term">Pipelines & overlap</span>: **1F1B** alternates microbatches forward/backward to keep stages busy and reduce bubbles; more aggressive schemes interleave MoE **all-to-all** under compute to hide communication.

* <span class="term">Precision & memory tactics</span>: **activation recompute** and **mixed precision** are standard to fit HBM; some layers may store activations at reduced precision while computing in higher precision.

* <span class="term">Topology awareness</span>: placement aims to keep hot token→expert paths **intra-node** on NVLink/NVSwitch and minimize **inter-node** RoCE/IB hops. DeepSeek extends this baseline with **node-limited routing** (covered later); K2 follows standard placement hygiene and leans on schedule/memory tactics.

* <span class="term">Shared-expert locality</span>: cache/pin the **shared expert** near the token path to avoid paying cross-node latency on common fall-back routes.


## 5. Auxiliary-Free Load Balancing (DeepSeek V3)
In our running picture, **lakes** represent regions of the data manifold and **fishermen** are the **experts**. A classic <span class="term">auxiliary diversity loss</span> is like a redistributive tax that penalizes popular lakes and subsidizes empty ones inside the **preference function** itself. This evens out utilization, but it also changes what “best lake” means for each fisherman and can reduce catch quality.

DeepSeek keeps preferences honest and instead changes the **reach** of each fisherman. Imagine walking along the shore with a pair of shears. For over-crowded lakes, you **trim the rod length** of the successful fishermen, shrinking their radius of influence. For under-served lakes, you **extend rod length**, expanding their reach. Preferences remain about fish density; availability is nudged externally. <span class="idea">Balance is achieved by shaping access, not by corrupting preference</span>.

TODO - check formatting of math blocks (square brackets vs $$)

**Math path vs control path.** Let (x) be a token representation and (e_i) expert (i).

* **Affinity (learning) path**
  [
  \ell_i = f(x, e_i) \quad\text{(e.g., a learned scorer or } q^\top k_i/\tau\text{)}
  ]
  Gradients for specialization flow through (\ell_i).

* **Control (utilization) path**
  Maintain a per-expert bias (b_i) updated from recent utilization (u_i) toward a target (u^*):
  [
  b_i \leftarrow \mathrm{EMA}\big(b_i + \alpha,(u^* - u_i)\big)
  ]
  The update can be viewed as a PI-like controller where the EMA provides the integral term. Importantly the bias is applied as **stop-gradient**:
  [
  s_i = \ell_i + \texttt{stopgrad}(b_i)
  ]
  Routing selects top-(k) by (s_i), but backprop only touches (\ell_i), not (b_i).

* **Sequence-level safeguard**
  A tiny sequence-wise stabilization term can be added to prevent collapse without altering per-token affinities.

This split keeps token→expert **affinity clean** while a slow control loop adjusts **effective availability**. In control terminology, the utilization error (u^* - u_i) drives a PI controller that trims or extends each expert’s “rod length.” The router continues to learn specialization from the true signal.



## 6. Other Innovations

* <span class="term">MLA attention</span>: latent KV compression that cuts memory and traffic, enabling longer contexts.
* <span class="term">MTP (DeepSeek)</span>: multi-token prediction objective that changes gradient pathways and interacts with routing frequency.
* <span class="term">MuonClip (K2)</span>: Muon optimizer plus **QK-Clip** (per-head logit rescale beyond a threshold τ), stabilizing long pretrain **[VERIFY ~15.5T tokens]** and preventing attention-logit blowups.
* <span class="term">Precision and memory</span>: FP8 activation storage, targeted recompute, and selective CPU offload (emphasized in K2) to fit deep PP/EP without starving bandwidth.
* <span class="term">Topology-aware kernels</span>: custom all-to-all and packing paths that reduce launch overhead and improve link utilization on NVLink and RoCE.
* <span class="term">Activation scheduling</span>: selective checkpointing and recompute placement to keep overlap feasible under fixed HBM budgets.
* <span class="term">KV cache tactics</span>: quantization and layout choices that shrink cache footprint while preserving attention quality.
* <span class="term">Inference path hygiene</span>: shard placement, batching, and cache locality strategies that keep hot paths intra-node for low-latency decode.
