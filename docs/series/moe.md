
# The Sparsity Frontier: Core Innovations in Kimi K2 and DeepSeek V3
(NOTE: UNDER CONSTRUCTION)

## 1. Introduction
A core design limitation of standard <span class="term">dense LLMs</span> is the tight coupling between parameter count ("knowledge"), and per-token flop count (the amount of "work" required to generate a token), due to knowledge being diffuse across parameters rather than localized.
When we ask "what is the capital of Kenya?", we effectively pay for the LLM's
proficiency in Medieval history, Korean literature, and quantum mechanics, despite the bounded scope
of our question.

A natural solution to this problem is <span class="idea">sparsity</span>: decouple memory and
compute by selectively activating a <span class="idea">bounded subset</span> of parameters for each
layer-token pair, thereby implicitly localizing knowledge. <span class="term">Mixture of Experts
(MoE)</span> layers
are a particular class of solutions in the design space of sparse architectures, and are the
standard approach used today. 

Despite its intuitive appeal, sparsity raises complex challenges: the imposition
of discrete structural constraints on systems we train with continuous tools (backprop) on
distributed hardware raises implications for modeling (<span class="idea">optimization dynamics</span>, <span class="idea">manifold geometry</span>), systems (increased inter-device <span class="idea">communication</span>, activation <span class="idea">memory pressure</span>, reduced <span class="idea">data locality</span>),
and post-training (potential <span class="idea">discrete shifts</span> under distributional drift). Progress in
this area hence necessitates expertise in modeling and systems alike. In this article, we'll first frame these core challenges, and then explore how bleeding edge sparse
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
post-training. In this article, we'll primarily focus on pretraining aspects directly connected to
MoEs and sparsity. In particular, we'll discuss:

* Challenges with training large sparse models, and frames for reasoning about them (Section 3)
* Core elements of Kimi K2 and Deepseek V3's sparsity architectures (Section 4)
* Several innovations in K2 and DV3 that address specific sparsity challenges (Section 5)
* A brief discussion of salient non-sparsity-specific aspects of these models (Section 6)
* A deeper dive into DV3's auxiliary-free load balancing (Part 2)


### 2.3 Notation
We consider MoE layers with input dimension <span class="term">$D$</span>,
<span class="term">$m$</span> total experts, and <span class="term">$k$</span> selected experts (per token, per layer). In addition, we'll let <span class="term">$k_f$</span> denote the number of
shared/fixed experts ($m$ and $k$ refer only to non-shared), and <span class="term">$T$</span> the
context length. (See the spec table in Section 4.3 for a full list of symbols we use across the article.) We'll use DV3 and K2 as shorthand for Deepseek V3 and Kimi K2. Key terms are highlighted
in <span class="term">blue</span>, and key ideas in <span class="idea">green</span>; hence a good
way to skim this article is to follow the colored words.

## 3. The Challenge of Sparsity
### 3.1 Systems Challenges - High Level Framing
A general design goal in modern pipeline-parallel distributed training is to hide latency by
scheduling compute precisely to overlap with communication; hence anything that increases
communication or makes it less predictable poses a challenge.

Sparsity allows flop count to scale sublinearly in parameter count, but those parameters still need
to live somewhere, so memory (and hence the number of devices) still grow linearly in parameter
count. Hence sparsity effectively shifts the burden from
<span class="idea">compute</span> to <span class="idea">memory</span> and inter-device <span class="idea">communication</span>. As we'll see, the resulting systems challenges are tightly
coupled to architectural choices.

### 3.2 Dynamic Dispatch
Expert parallelism shards the $m$ experts across $d$ devices. The choice of which expert processes
which token is a function of the token, tokens must be
<span class="term">dynamically dispatched</span> to experts. Mechanically, this involves packing tokens into contiguous <span class="term">send
buffers</span> (with padding for aligmment), exchange via <span class="term">all-to-all dispatch</span>,
processing as small <span class="term">GEMM</span> batches, and finally unpacking to the original order via
<span class="term">all-to-all reduction</span>. This communication pattern raises several potential
issues, including imbalanced loads, fragmented memory access, scattered gathers, small kernels, and
poor cache locality. 

### 3.3 Intra-Node vs Inter-Node Communication
GPUs in training clusters are organized into <span class="term">nodes</span>. A node is typically a
single server containing multiple GPUs (often 8 or 16), and GPUs in the same node communicate via
<span class="term">NVLink</span> or <span class="term">NVSwitch</span>, while communication across
nodes uses <span class="term">InfiniBand</span> or <span class="term">RoCE</span>. These have very
different bandwidths: NVLink provides up to ~1.8 TB/s per GPU on Blackwell (bidirectional, per GPU),
while InfiniBand offers ~100 GB/s per port (XDR 800), typically aggregated over multiple ports per
node. The implication for model designers is that communication costs are <span
class="idea">heterogeneous</span>, which motivates thinking of experts not just in isolation, but
potentially defining <span class="idea">topology-aware groupings</span> of experts based on physical
colocation. We'll see in Sections 4.1 and 5.2 how <span class="term">dispersion bounding</span> and
<span class="term">hot expert replication</span> are concrete instances of this idea.

### 3.4 Memory Pressure
In MoE layers, gradients don't flow to non-selected experts, so at first glance, it should seem that
like compute, the amount of state we need to persist for the backward pass (besides weights
themselves) scales with $k$, not $m$. A few issues complicate this picture. 

<span class="term">Routing Metadata</span><br>
Routing necessitates a lot of bookkeeping: top-k indices, permutation maps, and scatter/gather
layouts to invert token dispatch for the backwards pass, plus moving averages/auxiliary terms for
load balancing. This is still $O(kT)$, but nontrivial overhead nonetheless. 

<span class="term">Comms Buffers</span><br>
Under expert parallelism, tokens are packed into per-destination fixed-capacity send/recv buffers.
Padding becomes significant under imbalanced loads, particularly early in training.

<span class="term">Optimizer State</span><br>
Modern learning algorithms like Muon or Adam require per-parameter additional state, which grows
with $m$ rather than $k$. As the K2 paper notes, "*after reserving space for
parameters, gradient buffers, and optimizer states, the remaining \[HBM\] is insufficient to hold
the full MoE activations.*"

We'll see in Section 5 how K2 and DV3 address memory and communication challenges via various
techniques including reduced precision, activation recomputation, CPU offloading, novel pipeline
schedules, caching, replication, and others.

### 3.5 Modeling Challenges: Expert Specialization, Manifold Partitioning, and a Fishing Analogy
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

## 4. Sparsity Architecture
In this section we'll examine the broad contours of the architectures of K2 and DV3.

### 4.1 Expert Selection
The high-level elements of K2 and DV3's MoE layer are fairly familiar: for each token, first compute
<span class="term">token-expert affinities</span> with a per-expert sigmoid, then apply <span class="term">top-k hard gating</span>, then normalize
selected experts' affinities into <span class="term">scores</span>. Both K2 and DV3 use one <span class="term">shared
expert</span> that is exempt from this scoring process (its score is fixed to 1). 

DV3 introduces two core twists: <span class="term">auxiliary-free load balancing</span> via a bias
term (discussed in Part 2) and <span class="term">dispersion bounding</span>, discussed below.

<span class="term">Dispersion Bounding (DV3)</span>

Dispersion bounding reduces inter-node communication by explicitly capping how many <span class="idea">nodes</span> a single token may
touch in an MoE layer. Ordinary MoE routing simply selects the experts with the top $k$ scores. DV3
constrains this selection so that the selected experts reside in <span class="idea">at most 4
nodes</span>. Concretely, say we have 8 active experts, the top 7 scoring experts live on nodes
$n_1, n_2, n_3, n_4$, and the 8th expert lives on a fifth node $n_5$. The 8th expert would be
dropped and replaced by the next-highest scoring expert that lives on one of $n_1, n_2, n_3, n_4$.

### 4.2 Ultra Sparse Design
DV3 uses $k = 8$, $m = 256$, $k_f = 1$ (8 active experts out of 256, plus one fixed), for a sparsity
ratio of $s = 32$. Not only is this a high ratio, it is also very <span class="idea">fine-grained</span> sparsity. K2 pushes even further, with $k = 8$, $m = 384$. The table below compares these
models to some of their contemporaries. 

| Model | $m$ (total experts) | $k$ (active experts) | $s$ (expert sparsity) | $P$ (Total Params) | $P_a$ (Active Params) |
| --- | ---: | ---: | ---: | ---: | ---: |
| <span class="term">Kimi K2</span> | 384 | 8 | 48.0 | **1.04T** | **32B** |
| <span class="term">DeepSeek-V3</span> | 256 | 8 | 32.0 | **671B** | **37B** |
| <span class="term">GPT-OSS-120B</span> | 128 | 4 | 32.0 | **117B** | **5.1B** |
| <span class="term">DeepSeek-V2</span> | 160 | 6 | ~26.7 | **236B** | **21B** |
| <span class="term">Qwen3-235B</span> | 128 | 8 | 16.0 | **235B** | **22B** |
| <span class="term">OLMoE</span> | 64 | 8 | 8.0 | **7B** | **1B** |
| <span class="term">DBRX</span> | 16 | 4 | 4.0 | **132B** | **36B** |
| <span class="term">Grok-1</span> | 8 | 2 | 4.0 | **314B** | **~78.5B** |
| <span class="term">Mixtral 8x22B</span> | 8 | 2 | 4.0 | **141B** | **~39B** |
| <span class="term">Llama 3.1 405B</span> | 1 | 1 | 1.0 | **405B** | **405B** |

TODO - switch transformer and other google models?

<span class="term">Sparsity Scaling Laws</span>

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

### 4.3 Spec Table
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
| **Routing control** | **Aux-loss-free dynamic bias** + lightweight **sequence-level** safeguard | **Simple top-k** (no grouping, no in-gate balancing) |
| **Dispersion control** | **Node-limited ($\leq 4$ nodes/token)** | **None explicit; implicit via low EP** |
| **Attention Mechanism** | **MLA** | **MLA** |
| **Attention Heads**  | **128** | **64** |
| **Parallelism Strategy** | **DualPipe** | **Interleaved 1F1B** |
| **Memory Optimizations** | **Recompute**, **Reduced Precision**, **CPU Offload** | **Recompute**, **Reduced Precision**, **CPU Offload** |

## 5. Communication & Memory Optimizations
### 5.1 Training Parallelism & Communication
Both models compose <span class="term">pipeline parallelism (PP)</span>, <span class="term">expert
parallelism (EP)</span>, and [ZeRO-1 data parallelism (DP)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/zero1_gpt2.html)</span>.

<span class="term">No Tensor Parallelism</span><br>
As we saw in Section 3, cross-node communication under expert parallelism unfavorably tips the
balance of communication and computation. A natural step to counteract this, particularly with
fine-grained sparsity, is to <span class="idea">remove tensor parallelism</span>. The DV3 paper
explicitly mentions removing tensor parallelism during training, and the K2 paper does not mention
using tensor parallelism.

<span class="term">DualPipe (DV3)</span><br>
DV3 also introduces a novel pipeline schedule called DualPipe, whose core idea is to carefully
overlap communication and computation
<span class="idea">within a paired forward-backward channel</span>. Specifically, DualPipe splits
each layer into substages:

* Forward: attention, dispatch, MLP, combine 
* Backwards: combine, MLP (weights), MLP (inputs), dispatch, attention (weights), attention (inputs)

Each stage maintains two in-flight parameter/gradient residencies so a forward channel and a
backward channel can run concurrently without blocking on the same weights. With careful reordering,
DualPipe overlaps nearly all communication (MoE + pipeline) under compute, as illustrated in the
figure below from the DV3 paper. Note that this comes at the cost of increased memory footprint due
to the weight replication.

(TODO - insert pipeline figure)

<span class="term">Interleaved 1F1B (K2)</span><br>
K2's authors cite DualPipe’s extra parameter and gradient memory footprint as prohibitive for
scaling to a trillion parameters, and stick to
[interleaved 1F1B](https://colossalai.org/docs/features/pipeline_parallel/), an existing method
in which each stage alternates one forward and one backward microbatch, with a single parameter
copy. 

Since K2 uses only 64 attention heads (compared to 128 in DV3), there is an increased need to
reduce expert-parallel communication in order for it not to dominate during 1F1B. K2 achieves this
by adopting "the smallest feasible EP parallelization strategy," partitioning experts across just
16 devices. Note that lower expert parallelism implies more experts
per GPU, which implicitly smoothes load (even if load is imbalanced across experts, it has a higher
probability of being balanced across GPUs, due to the law of large numbers).

### 5.2 Hot Expert Replication (DV3)
The basic idea of replication is that during inference, we can monitor online statistics of expert
loads, and <span class="idea">redundantly deploy</span> high-load experts in a manner that balances
load across GPUs without increasing inter-node communication. DV3 applies this strategy to the
prefilling stage of inference specifically. It uses 32 redundant experts (out of 256 total), with
each GPU hosting its 8 original experts plus 1 redundant expert.

### 5.3 Custom Kernels
DV3 develops custom kernels using [PTX](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/) for efficient all-to-all communication. While
specifics of this kernel are beyond the scope of this article, what stands out is the
<span class="idea">extent of codesign</span>: the routing mechanism (dispersion bounding), cluster
topology, pipeline schedule (DualPipe), and custom kernels are all jointly optimized.

### 5.4 Memory Optimizations
<span class="term">Activation Recomputation</span><br>
Activation recomputation is a standard idea in pretraining, wherein certain high-memory, low-compute
layers are <span class="idea">recomputed</span> during the backwards pass rather than persisting
their activations, effectively trading a little compute overhead for large memory savings. This is
<span class="idea">particularly valuable for MoEs</span>, because:

* MoEs by design have a high memory to compute ratio
* Expert load-imbalance during early training could otherwise cause OOM crashes if we persisted
all activations

K2 uses aggressive activation recomputation, applying it to LayerNorm, SwiGLU, MLA up-projections,
and MoE down-projections. DV3 applies activation recomputation to RMSNorm and MLA up-projections. 

<span class="term">CPU Offloading</span><br>
The basic idea of CPU offloading is to move pieces of state computed on GPUs over to CPU RAM (or
even compute them entirely on CPUs), where possible.

For activations that are not recomputed, K2 offloads them to CPU RAM, using a custom streaming
copy engine that overlaps with both compute and communication kernels in the 1F1B schedule.

DV3 maintains an exponential moving average (EMA) of model parameters during training. Rather than
storing these in GPU memory, these are stored in CPU memory, and <span class="idea">updated asynchronously</span>.

<span class="term">KV Cache Reduction</span><br>
Both DV3 and K2 reduce KV cache memory footprint via Multi-Head Latent Attention (MLA),
discussed in the next subsection.

<span class="term">Reduced Precision</span><br>
Both DV3 and K2 make extensive use of reduced precision. This is a large topic in its own right,
and we'll leave a detailed treatment to a future article. 

## 6. Non-MoE Techniques
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

## 7. Conclusion and Future Directions
TODO


