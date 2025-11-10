
# The Sparsity Frontier: Core Innovations in Kimi K2 and DeepSeek V3
(UNDER CONSTRUCTION)

## 1. Introduction
The world spends (TODO - look up amount) per year on LLM inference, and demand is growing
exponentially (TODO - placeholder; replace hook if needed). A core design limitation of standard <span
class="idea">dense</span> LLMs is the tight coupling between parameters count (how much the LLM 
"knows") and flop count (how much "work" the LLM has to do to generate a token). When you ask a powerful LLM "what is the capital of Kenya?",
you are in some sense paying for the fact that the LLM is fluent in Korean,
an expert in Medieval history, and proficient in group theory, despite the simple and focused
nature of your question. More precisely, a standard MLP layer in a transformer with embedding
dimension $D$, and MLP width $wD$ has $O(D^2w)$ parameters and requires $O(CD^2w)$ flops to generate
$C$ tokens, meaning flop count scales linearly with parameter count.

A natural solution to this problem is <span class="idea">sparsity</span>: decouple memory and compute
by selectively involving a small subset of parameters in each token-generation step.
The standard way this is done today is through Mixture of Experts (MoE) layers. 

The core challenge of sparsity in general and MoEs in particular is that the imposition of discrete
structure on systems we train with continuous tools (backprop) creates a fundamental tension, with
implications for modeling, optimization, and systems. Moreover, these problems are tightly coupled;
hence progress in this direction necessitates expertise of modeling and systems alike. In this
article, we'll frame these core challenges, and explore how innovations in the latest Kimi and
Deepseek models address them. 


## 2. Prerequisites & Scope
This is a fairly advanced article, and we'll assume familiarity with the fundamentals of MoEs and
distributed training (e.g. 4D parallelism, ZeRO, etc.). We'll draw on some ideas from our previous article on [transformers as information flow graphs](TODO - link); while
this isn't strictly required pre-reading, we recommend skimming section 3. Our focus in this article
will be on MoE innovations; topics like sparse attention, reduced precision training, etc. will just
be briefly touched on in Section 7. 

The table below standardizes the notation we'll be using throughout. 

TODO (@ LLM) - put this in a table
D_1: embedding dimension
m: total number of experts
k: active experts (not counting fixed experts)
k_f: fixed experts
b: expert size
D_2: MoE hidden dimension (D_2 = b*m)

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

### 3.2 Modeling Challenges: Manifold Partitioning, Specialization, and Dead Experts
TODO - this part needs substantial changes
The promise of MoE rests on <span class="term">expert specialization</span>. Specialization fails when <span class="term">dead experts</span> emerge from gradient starvation, when a few experts monopolize traffic, or when routing boundaries jitter so much that experts cannot settle on stable roles. Classic <span class="term">auxiliary load-balancing losses</span> spread utilization but can corrupt token–expert affinities and weaken specialization. Temperature, entropy, and EMA schedules trade exploration against lock-in. Capacity limits force second-best choices when the top expert is full, which can help exploration but also increase variance. A <span class="term">shared expert</span> can backstop quality, yet too much shared capacity reduces pressure to specialize. Curriculum and post-training shift the data manifold, moving Voronoi-like boundaries and destabilizing routes unless the router adapts.

These modeling choices are not free of systems constraints. If we cap a token to ≤ <span class="term">M</span> nodes to control dispersion, routing becomes a constrained assignment problem. Scores must be aggregated by node, nodes are selected under the cap, and experts are filled within those nodes to reach top-k. Stronger balancing stabilizes throughput but risks misrouting. Larger expert pools improve coverage but add dispatcher metadata and increase small-message traffic. <span class="idea">The practical objective is the best expert assignment given a communication and
capacity budget, with gradients that remain faithful to token–expert affinity</span>.


---
## 4. Architecture Specs and Standard Elements

### 4.1 Spec Table


| Dimension                         | DeepSeek V3                                                                | Kimi K2                                                           |
| --------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Total Params**                  | **[VERIFY ~671B]**                                                         | **[VERIFY ~1.0T]**                                                |
| **Active Params (per token)**     | **[VERIFY ~37B]**                                                          | **[VERIFY ~32–33B]**                                              |
| **Total Experts**                 | **256 routed + 1 shared**                                                  | **384 routed + 1 shared**                                         |
| **Active Experts / Token**        | **8 routed** + shared always available **[VERIFY]**                        | **8 routed** + shared available                                   |
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


## 5. Auxiliary-Free Load Balancing (Deepseek V3)
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

---

## 6. Bounded Dispersion
TODO - this section feels too long


### 6.1 Why dispersion matters
TODO - reword
Inter-node communication is the expensive part of MoE. If a token’s activated experts live on many nodes, route variance turns into stragglers and small-message all-to-all. 

The useful abstraction is the token’s **communication subgraph**. <span class="term">Node-limited routing</span> makes this subgraph compact by capping how many nodes a token may touch at (M). <span class="idea">Treat dispersion as a first-class budget alongside top-(k)</span>.

### 6.2 A single-pass, effectively hierarchical algorithm
TODO - revise notation

Let there be (G) nodes and routed experts (E=\bigcup_{n=1}^{G} E_n) where (E_n) are the experts hosted on node (n). For a token with representation (x):

1. **Score experts.** Compute per-expert logits once
   [
   \ell_i = f(x, e_i), \quad i \in E.
   ]

2. **Score nodes.** Estimate which nodes could serve the token well using their strongest candidates. A simple, effective choice is a **top-(r)** aggregate
   [
   A_n = \sum_{i \in \text{Top}*r(E_n)} \ell_i,
   ]
   with (r) chosen so that (rM \ge k). A soft alternative is (\mathrm{lse}*{i\in E_n}(\ell_i)). The goal is the same: approximate a node’s headroom without another forward pass.

3. **Select nodes.** Choose up to (M) nodes with the largest (A_n).

4. **Select experts within chosen nodes.** From (\bigcup_{n\in \text{chosen}} E_n), pick the global top-(k) experts by (\ell_i), enforcing per-expert capacity.

The router still runs a single scoring pass over experts. The group selection over nodes makes it **effectively hierarchical** at node granularity. If one node already contains enough strong experts the token lands entirely on that node. Otherwise it touches at most (M) nodes by construction.

### 6.3 Effects on learning geometry
TODO - tighten

Capping dispersion shrinks the candidate set the router searches. That reduces cross-node route churn and smooths Voronoi boundaries induced by placement. Experts on the same node compete more often with one another, which encourages **local specialization** and reduces interference from far-flung candidates that would be costly to use. There is a tradeoff. If (M) is too small relative to how experts are distributed, the cap can force second-best assignments and slightly raise variance for tokens whose ideal experts are scattered. In practice this is mitigated by placement and a strong shared expert that backstops quality when the cap binds.

### 6.4 Systems consequences and why schedules get easier

Bounding dispersion turns a fuzzy risk into a predictable envelope.

* **Cross-node edges.** A token now contributes at most (M) cross-node edges in all-to-all. Incast width is smaller and more uniform. Straggler variance drops.
* **Overlap.** With narrower, repeatable all-to-all blocks, aggressive overlap is feasible. DualPipe-style timelines can nest MoE communication under forward and backward compute more reliably because long-tail sprays are pruned.
TODO - wtf is this placement bullet?
* **Placement.** With an (M) budget, placement pays off. Co-locate frequently co-activated experts. Keep the <span class="term">shared expert</span> proximate or cached so the default path remains intra-node.
* **Capacity planning.** Because the subgraph is compact, batch sizing and microbatching are easier to tune. Fewer small packets cross node boundaries which reduces launch overhead and better utilizes links.

<span class="idea">The modeling cap on dispersion is the enabler for a simpler and more stable communication schedule</span>.

### 6.5 Knobs
TODO - i removed existing stuff from this subsection because it looked totally speculative. check
actual knobs used in model


## 7. Finer-grained sparsity in Kimi

**Opener.** K2 increases the routed-expert **pool** to **[VERIFY: 384]** while keeping **active experts per token = 8** and a single <span class="term">shared expert</span> **[VERIFY]**. Total vs active parameters are roughly **[VERIFY: ~1.0T / ~32–33B]**.

### 7.1 Geometry and learning dynamics

With many more experts at fixed (k), the representation manifold is cut into **more, smaller cells**. Tokens search a denser set of nearby specialists at the same FLOP budget. Local competition rises among neighbors on the manifold, reducing interference from distant experts and letting features sharpen within each cell. The boundary grid becomes finer, which improves long-tail coverage but increases sensitivity to early traffic. Warm-up and temperature schedules matter so tokens explore enough candidates before settling. The <span class="term">shared expert</span> provides a floor when a niche cell is under-trained. <span class="idea">Breadth raises the odds of a near match without raising per-token compute</span>.

### 7.2 Costs—and how K2 pays them

A larger pool raises **routing entropy**, **dispatcher metadata**, and the number of **small messages**. It also risks <span class="term">expert under-training</span> if traffic is spiky. K2 absorbs these taxes with targeted systems choices:

* **Interleaved <span class="term">1F1B</span>** keeps stages busy and tucks MoE <span class="term">all-to-all</span> under compute on RoCE-heavy clusters, avoiding heavier orchestration.
* **Activation budget engineering** — <span class="term">selective recompute</span>, <span class="term">FP8 activation storage</span>, and <span class="term">CPU offload</span> with overlapped copies — reduces HBM pressure so EP traffic fits under compute.
* **Placement hygiene** spreads load and avoids hot spots so unconstrained dispersion does not explode into pathological fan-out.

Despite the finer granularity, K2 keeps the **router simple**: standard top-(k) over routed experts, no extra balancing in the gate. This works because the **systems margin** and **training stability** carry the complexity that V3 addresses in the router.



## 8. Non-MoE Innovations

* **MLA attention:** latent KV compression; reduces memory/traffic—enabler for long contexts.
* **MTP (DeepSeek):** multi-token prediction; interacts with routing frequency and gradient flow.
* **MuonClip (K2):** **Muon** optimizer + **QK-Clip** (per-head logit rescale past threshold τ) — stabilizes long pretrain **[VERIFY ~15.5T tokens]**; prevents attention-logit blowups.
* **Precision & memory:** **FP8 activation storage**, targeted **recompute**, and **CPU offload** (K2 emphasis) to fit deep PP/EP without starving bandwidth.

---


## 9. Open Questions & Future Directions

TODO