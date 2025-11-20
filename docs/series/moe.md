
# The Sparsity Frontier: Core Innovations in Kimi K2 and DeepSeek V3
(NOTE: UNDER CONSTRUCTION)

## 1. Introduction
A core design limitation of standard <span class="term">dense LLMs</span> is the tight coupling between parameter count ("knowledge") and per-token FLOP count (the amount of "work" required to
generate a token), due to knowledge being diffuse across parameters rather than localized.
When we ask "what is the capital of Kenya?", we effectively pay for the LLM's
proficiency in Medieval history, Korean literature, and quantum mechanics, despite the bounded scope
of our question.

A natural solution to this problem is <span class="idea">sparsity</span>: selectively activate a
<span class="idea">bounded subset</span> of parameters for each token, thereby decoupling memory and
compute and implicitly localizing knowledge. <span class="term">Mixture of Experts
(MoE)</span> layers are a form of structured sparsity, and are the standard approach used today. 

Despite its intuitive appeal, sparsity raises complex challenges: the imposition
of discrete structural constraints on systems we train with continuous tools (backprop) on
distributed hardware originally built for dense matrix math raises implications for modeling (<span class="idea">optimization dynamics</span>, <span class="idea">manifold geometry</span>), systems (increased inter-device <span class="idea">communication</span>, <span class="idea">memory pressure</span>), and post-training (potential <span class="idea">discrete shifts</span> under distributional drift). Progress in
this area hence necessitates advances in modeling and systems alike.

In this article, we'll first frame these core challenges, and then explore how bleeding-edge sparse
models <span class="term">[DeepSeek V3](https://arxiv.org/abs/2412.19437)</span> and
<span class="term">[Kimi K2](https://arxiv.org/abs/2507.20534)</span> address them while advancing
the frontier of sparsity.

---

## 2. Preliminaries
### 2.1 Prerequisites
This is a technical article. We'll assume familiarity with the basics of what
[MoE layers](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) are
and how [distributed training](https://colossalai.org/docs/concepts/paradigms_of_parallelism) works.

### 2.2 Scope & Roadmap
Kimi K2 and DeepSeek V3 introduce a number of innovations across data, pretraining, and 
post-training. We'll primarily focus on pretraining aspects directly connected to
MoEs and sparsity. In particular, we'll discuss:

* Challenges with training large sparse models, and frames for reasoning about them (Sections 3 and 4)
* Core elements of Kimi K2 and Deepseek V3's sparsity architectures (Section 5)
* Several innovations in K2 and DV3 that address specific sparsity challenges (Sections 6-8)

This article will mostly focus on systems innovations and systems-modeling codesign. We'll cover
a very interesting pure modeling piece (auxiliary-free load balancing in Deepseek V3) in Part 2.

### 2.3 Notation
We consider MoE layers with input dimension <span class="term">$D$</span>,
<span class="term">$m$</span> total experts, and <span class="term">$k$</span> selected experts (per token, per layer). In addition, we'll let <span class="term">$k_f$</span> denote the number of
shared/fixed experts ($m$ and $k$ refer only to non-shared), and <span class="term">$T$</span> the
context length. (See the spec table in Section 4.3 for a full list of symbols we use across the article.) We'll use DV3 and K2 as shorthand for Deepseek V3 and Kimi K2. Key terms are highlighted
in <span class="term">blue</span>, and key ideas in <span class="idea">green</span>; hence a good
way to skim this article is to follow the colored words.

---

## 3. Systems Challenges for Sparsity
### 3.1 High Level Framing
A general design goal in modern distributed training is to hide latency by scheduling compute
to overlap with communication; hence anything that increases communication or makes it less
predictable poses a challenge.

Sparsity lets FLOP count sublinearly with parameter count, but parameters still need to live
somewhere, so total memory - and hence device count - scales roughly linearly with parameters.
Sparsity effectively shifts the burden from
<span class="idea">compute</span> to <span class="idea">memory</span> and
<span class="idea">communication</span>, with negative implications for
[arithmetic intensity](https://modal.com/gpu-glossary/perf/arithmetic-intensity) if not
mitigated. As we'll see, the resulting systems challenges are tightly coupled to architectural
choices, meaning model designers cannot simply abstract away infrastructure details, resulting in a rich surface area for infrastructure-architecture codesign. 

### 3.2 Dynamic Dispatch
<span class="term">Expert parallelism</span> partitions the $m$ experts across $d$ devices. The
choice of which expert processes which token is a function of the token, so tokens must be
<span class="term">dynamically dispatched</span> to experts. Mechanically, this involves packing tokens into contiguous per-destination <span class="term">send
buffers</span> (with padding for alignment), exchange via <span class="term">all-to-all dispatch</span>,
processing as <span class="term">batched small GEMMs</span>, and finally another <span class="term">all-to-all</span> to unpermute to the original order. Communication is
overlapped with computation via <span class="term">double buffering</span>.

This communication pattern raises several potential issues, including imbalanced loads, fragmented
memory access, scattered gathers, small kernels, and poor cache locality.

### 3.3 Intra-Node vs Inter-Node Communication
GPU training clusters are composed of <span class="term">nodes</span>, each of which is a server
typically containing 8-16 GPUs. Within a node, GPUs communicate via
<span class="term">NVLink</span> or <span class="term">NVSwitch</span>, whereas communication across
nodes uses <span class="term">InfiniBand</span> or <span class="term">RoCE</span>. These have very
different bandwidths: NVLink provides up to ~1.8 TB/s per GPU on Blackwell (bidirectional, per GPU),
while InfiniBand offers ~100 GB/s per port (XDR 800), typically aggregated over multiple ports per
node. The implication for model designers is that communication costs are <span
class="idea">heterogeneous</span>. This motivates thinking of experts not just in isolation, but
potentially defining <span class="idea">topology-aware groupings</span> of experts based on physical
colocation. We'll see in Sections 5.1 and 6.3 how <span class="term">dispersion bounding</span> and
<span class="term">hot expert replication</span> are concrete instances of this idea.

### 3.4 Memory Pressure
In MoE layers, gradients don't flow to non-selected experts, so at first glance, it should seem that
like compute, the amount of state we need to persist for the backward pass (besides weights
themselves) scales with $k$, not $m$. A few issues complicate this picture. 

<span class="term">Routing Metadata</span><br>
Routing necessitates a lot of bookkeeping: top-k indices, permutation maps, and scatter/gather
layouts to invert token dispatch for the backward pass, plus auxiliary terms for load balancing.
This is still $O(kT)$ per layer, but nontrivial overhead nonetheless. 

<span class="term">Comms Buffers</span><br>
Padding in send/recv buffers becomes significant under imbalanced loads and small per-expert
batches, especially early in training; double buffering increases the peak footprint.

<span class="term">Optimizer State</span><br>
Algorithms like Muon and Adam require per-parameter additional state, which grows with $m$ rather
than $k$, and is typically FP32 even if weights are reduced precision. The K2 paper notes, "*after
reserving space for parameters, gradient buffers, and optimizer states, the remaining \[HBM\] is
insufficient to hold the full MoE activations.*"

<span class="term">Load Imbalance</span><br>
In addition to *total* memory, load imbalances can tip high-load GPUs over their individual limits, causing OOM crashes.

We'll see in Sections 5-8 how K2 and DV3 address memory and communication challenges via various
techniques including dispersion bounding, novel pipeline schedules, replication, caching, reduced
precision, activation recomputation, CPU offloading, and others.

---

## 4. Modeling Challenges: Expert Specialization, Manifold Partitioning, and a Fishing Analogy (Optional Section)
Intuitively, a modeling design goal of MoE layers is for different experts to specialize to "cover"
different parts of the input data manifold. A few potential failure modes include:

* Under-specialization (several experts learning the same thing) 
* "Dead experts" (some experts never getting selected by the router)
* Load imbalance (some experts activating far more frequently than others)

Below we describe a toy analogy for thinking about how these problems can arise under vanilla top-k
routing (without interventions like load balancing regularizers, capacity limits, etc.), and hence
motivate those interventions.

Imagine we have several lakes (regions of the data manifold), each with varying number of fish (data
samples). We need to allocate fishermen (experts) to these lakes, under competing constraints.

To start, say we have just 2 lakes, with 10 and 4 fish respectively, and 2 fishermen. Allocating
both fishermen to the lake with 10 fish is globally suboptimal (10 fish caught vs 14), but locally
optimal for each fisherman (5 fish each vs 4 if they switch), with no incentive (gradient) to switch
to the uncovered lake. A third fisherman, who starts in a barren lake with no fish (bad expert
initialization), starves (zero gradient) rather than switching to the untapped 2nd lake. A fourth, who
discovers a populated lake with 100 fish, becomes disproportionately wealthy (load imbalance),
without any redistributive mechanism (gradients under hard top-k gating creating a "rich get richer" phenomenon).

These issues make training MoEs tricky, necessitating careful auxiliary losses/regularizers
to ensure proper specialization and load balancing, plus monitoring during training to detect and
revive dead experts. However, we'll see in Part 2 how DV3 was able to dispense with (most of)
these auxiliary losses via a novel algorithm.

---

## 5. Sparsity Architecture
In this section we'll examine the broad contours of the architectures of K2 and DV3.

### 5.1 Expert Selection
The high-level elements of K2 and DV3's MoE layer are fairly familiar: for each token, first compute
<span class="term">token-expert affinities</span> with a per-expert sigmoid, then apply <span class="term">top-k hard gating</span>, then normalize
selected experts' affinities into <span class="term">scores</span>. Both K2 and DV3 use one <span class="term">shared
expert</span> that is exempt from this scoring process (its score is fixed to 1). 

DV3 introduces two core twists: <span class="term">auxiliary-free load balancing</span> via a bias
term (discussed in Part 2) and <span class="term">dispersion bounding</span>.

<span class="term">Dispersion Bounding (DV3)</span><br>
Dispersion bounding reduces inter-node communication by explicitly capping how many <span class="idea">nodes</span> a single token may
touch in an MoE layer. Ordinary MoE routing simply selects the experts with the top $k$ scores. DV3
constrains this selection so that the selected experts reside in <span class="idea">at most 4
nodes</span>. Concretely, say we have 8 active experts, the top 7 scoring experts live on nodes
$n_1, n_2, n_3, n_4$, and the 8th expert lives on a fifth node $n_5$. The 8th expert would be
dropped and replaced by the next-highest scoring expert that lives on one of $n_1, n_2, n_3, n_4$.

### 5.2 Ultra Sparse Design
DV3 uses $k = 8$, $m = 256$, $k_f = 1$ (8 active out of 256, 1 fixed), for a sparsity
ratio of $s = 32$. K2 pushes further, with $k = 8$, $m = 384$. This is not only a high ratio, but
also very <span class="idea">fine-grained</span> sparsity. The table below compares these
models to some of their contemporaries whose specs are public. 

| Model | Year | $m$ (total experts) | $k$ (active experts) | $s$ (expert sparsity) | $P$ (Total Params) | $P_a$ (Active Params) |
| --- | ---: | ---: | ---: | ---: | ---: |
| <span class="term">Kimi K2</span> | 2025 | 384 | 8 | 48.0 | **1.04T** | **32B** |
| <span class="term">DeepSeek-V3</span> | 2024 | 256 | 8 | 32.0 | **671B** | **37B** |
| <span class="term">Qwen3-235B</span> | 2025 | 128 | 8 | 16.0 | **235B** | **22B** |
| <span class="term">GPT-OSS-120B</span> | 2025 | 128 | 4 | 32.0 | **117B** | **5.1B** |
| <span class="term">DeepSeek-V2</span> | 2024 | 160 | 6 | 26.7 | **236B** | **21B** |
| <span class="term">Switch-C</span> | 2021 | 2048 | 1 | 2048.0 | **1.57T** |  **13B** |
| <span class="term">OLMoE</span> | 2024 | 64 | 8 | 8.0 | **7.0B** | **1.3B** |
| <span class="term">DBRX</span> | 2024 | 16 | 4 | 4.0 | **132B** | **36B** |
| <span class="term">Grok-1</span> | 2024 | 8 | 2 | 4.0 | **314B** | **~86B** |
| <span class="term">Mixtral 8x22B</span> | 2024 | 8 | 2 | 4.0 | **141B** | **~39B** |
| <span class="term">Llama 3.1 405B</span> | 1 | 1 | 1.0 | **405B** | **405B** |

K2 and DV3 push sparsity further than all their contemporaries, besides Google's Switch Transformer
(Switch-C), which was ahead of its time and an outlier in every dimension in this table. My sense is
that the representational weaknesses of $k=1$ outweigh the systems simplifications it brings, and
the industry has moved towards $k$ in the 2-8 range. I would not be surprised if future models
actually <span class="idea">raise</span> $k$, and push $s$ by raising $m$ and decreasing expert
width $b$, i.e. pursuing finer grained sparsity rather than just fewer active experts. We'll defer
a deeper discussion of the representational implications of this to a future article.

### 5.3 Sparsity Scaling Laws
The K2 paper develops an empirical <span class="idea">Sparsity Scaling Law</span>, in which they
observe:

"*Under a fixed number of activated parameters (i.e., constant FLOPs) — increasing the total number
of experts \[...\] consistently lowers both the training and validation loss \[...\]. Concretely,
under the compute-optimal sparsity scaling law, achieving the same validation loss of 1.5, sparsity
48 reduces FLOPs by 1.69x, 1.39x, and 1.15x compared to sparsity levels 8, 16, and 32,
respectively. Though increasing sparsity leads to better performance, this gain comes with increased infrastructure complexity.*"

These empirical findings corroborate our intuition that sparsity <span class="idea">makes sense
computationally</span>, and that the primary bottlenecks come from today's infrastructure. With
novel hardware and algorithms, might we see models with 1000x or 10000x sparsity in the
not-so-distant future?

### 5.4 Spec Table
The table below summarizes several key aspects of DV3 and K2's architectures.

| Dimension | Deepseek V3 | Kimi K2 |
| --- | --- | --- |
| **Total Model Params ($P$)** | **[671B]** | **[1.04T]** |
| **Active Model Params ($P_a$)** | **[37B]** | **[32.6B]** |
| **Total:Active Param Ratio** | **18.3** | **31.9** |
| **Pretraining Tokens** | **14.8T** | **15.5T** |
| **Total Layers ($L$)** | **61** | **61** |
| **Embedding Dimension ($D$)** | **7168** | **7168** |
| **Context Length ($T$)**  | **128K**  | **128K** |
| **Total Experts ($m$)** | **256** | **384** |
| **Active Experts ($k$)** | **8** | **8** |
| **Expert Sparsity ($s$)** | **32** | **48** |
| **Shared Experts ($k_f$)**| **1** | **1** |
| **Expert width ($b$)** | **2048** | **2048** |
| **Learning algorithm** | **AdamW** | **MuonClip** |
| **Routing control** | **Aux-loss-free dynamic bias** + lightweight **sequence-level** safeguard | **Simple top-k** (no grouping, no in-gate balancing) + **standard aux losses** |
| **Dispersion control** | **Node-limited ($\leq 4$ nodes/token)** | **None explicit; implicit via low EP** |
| **Attention Mechanism** | **MLA** | **MLA** |
| **Attention Heads**  | **128** | **64** |
| **Parallelism Strategy** | **DualPipe** | **Interleaved 1F1B** |
| **Memory Optimizations** | **Recompute**, **Reduced Precision**, **CPU Offload** | **Recompute**, **Reduced Precision**, **CPU Offload** |

---

## 6. Communication Optimizations
### 6.1 Forms of Parallelism
Both models compose <span class="term">pipeline parallelism (PP)</span>, <span class="term">expert
parallelism (EP)</span>, and [ZeRO-1 data parallelism (DP)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/zero1_gpt2.html)</span>.

As we saw in Section 3, cross-node communication under expert parallelism unfavorably tips the
balance of communication and computation. A natural step to counteract this, particularly with
fine-grained sparsity, is to <span class="idea">remove tensor parallelism</span>. The DV3 paper
explicitly mentions removing tensor parallelism during training, and the K2 paper does not mention
using tensor parallelism.

### 6.2 Pipeline Schedules
<span class="term">DualPipe (DV3)</span><br>
DV3 introduces a novel pipeline schedule called DualPipe, whose core idea is to carefully
overlap communication and computation
<span class="idea">within a paired forward-backward channel</span>. Specifically, DualPipe splits
each layer into substages:

* Forward: attention, dispatch, MLP, combine 
* Backward: same, but attention and MLP further split into `dInput` and `dWeight`

Each stage maintains two in-flight parameter/gradient residencies so a forward channel and a
backward channel can run concurrently without blocking on the same weights. With careful reordering,
DualPipe overlaps nearly all communication (MoE + pipeline) under compute, as illustrated in the
figure below from the DV3 paper. Note that this comes at the cost of increased memory footprint due
to the weight replication.

![pipeline_figure](../img/post1/moe_pipeline.png)

<span class="term">Interleaved 1F1B (K2)</span><br>
K2's authors cite DualPipe’s extra parameter and gradient memory footprint as prohibitive for
scaling to a trillion parameters, and stick to
[interleaved 1F1B](https://colossalai.org/docs/features/pipeline_parallel/), an existing method
in which each stage alternates one forward and one backward microbatch. Unlike DualPipe, 1F1B does
not require an extra copy of parameters.

Since K2 uses only 64 attention heads (compared to 128 in DV3), there is an increased need to
reduce expert-parallel communication in order for it not to dominate during 1F1B. K2 achieves this
by adopting "the smallest feasible EP parallelization strategy," partitioning experts across just
16 devices. Note that lower expert parallelism implies more experts
per GPU, which implicitly smoothes load (even if load is imbalanced across experts, it has a higher
probability of being balanced across GPUs, due to the law of large numbers).

### 6.3 Inference: Hot Expert Replication (DV3)
The basic idea of replication is that during inference, we can monitor online statistics of expert
loads, and <span class="idea">redundantly deploy</span> high-load experts in a manner that balances
load across GPUs without increasing inter-node communication. DV3 applies this strategy to the
prefilling stage of inference specifically. It uses 32 redundant experts (out of $m=256$ total),
with each GPU hosting its 8 original experts plus 1 redundant expert.

### 6.4 Custom Kernels
DV3 develops custom kernels using [PTX](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/) for efficient all-to-all communication. While
specifics of this kernel are beyond the scope of this article, what stands out is the
<span class="idea">extent of codesign</span>: the routing mechanism (dispersion bounding), cluster
topology, pipeline schedule (DualPipe), and custom kernels are all jointly optimized.

---

## 7. Memory Optimizations
### 7.1 Activation Recomputation
Activation recomputation is a standard idea in pretraining, wherein certain high-memory, low-compute
layers are <span class="idea">recomputed</span> during the backward pass rather than persisting
their activations, effectively trading a little compute overhead for large memory savings. This is
<span class="idea">particularly valuable for MoEs</span>, because they have a high memory to
compute ratio by design, and because they are particularly vulnerable to OOMs due to load imbalance.

K2 uses aggressive activation recomputation, applying it to LayerNorm, SwiGLU, MLA up-projections,
and MoE down-projections. DV3 applies activation recomputation to RMSNorm and MLA up-projections. 

### 7.2 CPU Offloading
The basic idea of CPU offloading is to identify pieces of state that can be computed on GPUs then
transferred to CPU RAM, or computed entirely on CPUs.

For activations that are not recomputed, K2 offloads them to CPU RAM, using a custom
<span class="term">streaming copy engine</span> that overlaps with both compute and communication kernels in the 1F1B schedule.

DV3 maintains an exponential moving average (EMA) of model parameters during training. Rather than
storing these in GPU memory, these are stored in CPU memory, and <span class="idea">updated asynchronously</span>.

### 7.3 Reduced Precision
Both DV3 and K2 make extensive use of reduced precision. This is a large topic in its own right,
and we'll leave a detailed treatment to a future article.

### 7.4 KV Cache Reduction
Both DV3 and K2 reduce KV cache memory footprint via Multi-Head Latent Attention (MLA),
briefly discussed in the next section.

---

## 8. Non-MoE Techniques
Below we highlight a few other significant architectural innovations in K2 and DV3, that are not
directly connected to sparsity but still highly influence overall efficiency.

<span class="term">Multi-Token Prediction (DV3)</span><br>
DV3 trains the model to the next two tokens simultaneously, as opposed to single next token
prediction. During inference, this prediction can be used for speculative decoding, enabling a
~1.8x TPS speedup in practice.

<span class="term">MLA (DV3 and K2)</span><br>
Both DV3 and K2 use <span class="term">Multi-head Latent Attention</span>, a novel attention
mechanism introduced by DV3. MLA factors attention through a lower-dim latent and caches this
latent during inference, cutting KV size and memory traffic without accuracy regressions.

<span class="term">MuonClip (K2)</span><br>
K2’s training stability hinges on <span class="term">MuonClip</span>, which augments the
[Muon](https://jeremybernste.in/writing/deriving-muon) algorithm with a <span
class="idea">QK-clip</span> to prevent exploding attention logits. The paper reports 15.5T
tokens of pretraining without loss spikes.

## 9. Reflections
TODO - complete
I started this article intending to write a short note on what I anticipated would be a few key
ideas. As I dove in, the extent of detail and innovation in the K2 and DV3 papers was quite
striking - about 10 pages of writing later and I feel like I've barely scratched the surface! I was
particularly struck by the extent of model-infrastructure codesign, and think this portends some
interesting things for the future.
