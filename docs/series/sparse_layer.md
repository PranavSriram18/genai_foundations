
# Rethinking Mixture of Experts through Sparse Coding and Competitive Learning

## 1. Introduction

Mixture of Experts (MoE) layers have become the dominant paradigm for decoupling parameter count
from compute in large transformer models. A common framing of MoEs is one of dynamic conditional
computation: we have a collection of experts, each an MLP, and we conditionally gate their outputs
based on a lightweight router. From that perspective, functional specialization emerges from the
router's partitioning of the input data manifold.

TODO - briefly explain core problems/shortcomings - expert collapse/redundancy, dead experts, etc.

Our goal in this work is to rethink the MoE setup from the perspective of sparse representations,
specifically leveraging principles from sparse coding and competitive learning. (TODO - add links).
Some particular frames we will develop include: 

* Each expert performs a low-rank read and a low-rank write, much like attention heads.
* In our setup, experts **compete** to explain an input by capturing “energy” from different directions in representation space.
* The goal is for experts to learn a set of near-orthogonal “views” of the residual stream that
specialize to different regions of the input manifold.
* Ultimately, what we're doing is learning a decomposition of the data manifold into combinations of
low dimensional subspaces.

This lens connects MoEs to classical sparse coding, competitive learning, and compressed sensing,
and suggests a set of design principles not visible if we think of MoEs purely as a form of
conditional computation.

TODO - clarify purpose of this article as framing/background, with our experiments coming in next
article.

We'll spell out this mental model more concretely in Section 2, and connect it to some classical
work. In Section 3, we'll situate some recent papers along these themes within the frames we've
developed. In Section 4, we'll expand on the specifics of the algorithm we have in mind as well as
various open questions. 

## 2. Core Frames

### 2.1 From dense MLPs to expert dictionaries

Consider a standard transformer block with residual stream dimension $D$. A dense MLP layer is a map

$x \in \mathbb{R}^D \quad \mapsto \quad x + U f(V^\top x)$

where $V \in \mathbb{R}^{D \times D_2}$ is the “read” matrix, $U \in \mathbb{R}^{D_2 \times D}$ is
the “write” matrix, $D_2$ is the hidden width, and $f$ is some nonlinearity.

An MoE layer does not fundamentally change this picture. It just **partitions** the hidden dimension into blocks that we call experts.

Let

* $m$ = number of experts
* $b$ = width of each expert
* $k$ = number of active experts
* $D_2 = m b$ = total hidden width

We can think of the hidden dimension as $m$ contiguous chunks (experts) of size $b$. Expert $i$ corresponds to:

* a read submatrix $V_i \in \mathbb{R}^{D \times b}$
* a write submatrix $U_i \in \mathbb{R}^{b \times D}$

If $S \subset {1, \dots, m}$ is the set of active experts for a given token, the layer implements

$x \quad \mapsto \quad x + \sum_{i \in S} U_i f(V_i^\top x)$

If we freeze the set $S$, this is just a **sum of low rank updates**. Each expert performs

$\mathbb{R}^D \xrightarrow{\text{read } V_i^\top} \mathbb{R}^b \xrightarrow{f} \mathbb{R}^b \xrightarrow{\text{write } U_i} \mathbb{R}^D$,

and these contributions add in the residual stream.

This “low rank read + write” view is the starting point. It suggests a different question than “how do we route tokens.” The question becomes:

How do we choose and train a set of low rank reads and writes so that
1. expert contributions are non-redundant, i.e. the read-subspaces of pairs of experts are largely
disjoint
2. for each input, a **sparse subset** of experts captures most of the energy?

### 2.2 Experts as lights casting shadows: energy capture

One analogy to illuminate the energy capture view is to imagine a 3D object in a room behind a
screen. You cannot see the object directly, and only see the shadows cast on the screen by different
lights.

* The object is the true state of the residual stream $x$.
* Each light corresponds to a read matrix $V_i$.
* The shadow on the screen is the hidden representation $h_i = V_i^\top x$.

Depending on the angle and position of a light, its shadow can either squash the object into an uninformative blob or preserve enough structure to recognize it.

In our setting, expert (i) “sees” the input only through (h_i = V_i^\top x). If (|h_i|_2^2) is small, that expert is effectively blind to this token. If (|h_i|_2^2) is large, the expert is capturing a big chunk of the **energy** of (x) along the directions it cares about.

This suggests a natural routing rule:

For a given token, pick the experts that capture the most energy in their reads.
More precisely, pick the top (k) experts by (|V_i^\top x|_2^2).

Instead of a separate router network, we directly select experts based on their relevance to the
input, as measured by the size of their read, which is the alignment between their subspace and the
input vector.

This is related to the intuition behind **k-means** and classical competitive learning:

* In the b = 1 limit, each expert reduces to a single direction (a centroid).
* In k-means, centroids compete for ownership of a data point, and each centroid moves to better represent the points it wins.
* Here, each expert is a low dimensional subspace rather than a single vector, and experts compete to capture energy from points in different regions of the manifold.


### 2.3 Geometry of reads: incoherence and compressed sensing

For the energy-capture competition setup to work, we need to address a couple natural failure modes.

First, each expert can increase the size of |V_i x| by simply scaling its norm. We can fix this by
constraining all columns of V to be unit norm.   

A slightly more subtle point is that experts can “cheat” by all pointing in the same high variance directions.

* Within an expert: each column of (V_i) can try to align with the top principal component of the data.
* Across experts: different (V_i) can duplicate each other.

That would maximize (|V_i^\top x|_2^2) for many tokens, but it defeats the whole point of an MoE. You end up with many copies of the same expert, not a diverse dictionary.



So we need geometric structure on the reads:

1. **Length control.** Columns of (V) should have controlled norm. Otherwise an expert can always inflate its energy by scaling up weights.
2. **Approximate orthogonality.** Directions within and across experts should be as close to orthogonal as the dimension allows, so that:

   * different experts are forced to specialize, and
   * sparse combinations of experts are well behaved.

Full orthogonality is impossible. There are (m b) columns in (V) but only (D) orthogonal directions in (\mathbb{R}^D). What we can ask for is **incoherence**: a large set of unit vectors where pairwise cosine similarity is bounded by some small (\mu).

This is exactly the setting studied in **compressed sensing** and sparse coding. A dictionary (V) that is roughly orthogonal on all small subsets behaves like an isometry when restricted to sparse codes. Formally, this is the **Restricted Isometry Property (RIP)**: for any subset (S) of columns with (|S| \leq k),

[
(1 - \delta)|c|_2^2 \le |V_S c|_2^2 \le (1 + \delta)|c|_2^2.
]

Translated back to MoEs:

* We want the combined read matrix (V) to be well conditioned on unions of a few experts.
* We can enforce this softly through regularizers that penalize deviations of (V_S^\top V_S) from the identity for small subsets (S).
* The experts are then pushed to occupy different “slots” in representation space, rather than piling onto the same directions.

The upshot is that a sparse MoE layer can be viewed as:

> A learned, overcomplete, approximately orthogonal dictionary (V),
> paired with a sparse, energy-based code that selects a few experts per token,
> and a write matrix (U) that maps those sparse codes back into the residual stream.

This is the conceptual frame. In the second half of the article we will spell out concrete architectural choices inside this frame. Before that, it is useful to look at how existing MoE and sparse-layer work fits into this picture.

## 3. How existing work fits into the sparse coding / competition lens

A lot of recent MoE work touches pieces of this story:

* competition between experts
* expert diversity and orthogonality
* representation collapse in routing
* sparse coding for interpretability

Most of it, however, treats these as isolated knobs rather than as consequences of a single geometric picture. In this section we situate a few representative papers inside the sparse coding / competitive learning lens described above, and highlight what they do, what they miss, and how they inform our direction.

We'll discuss the following papers:

* *Sparse Mixture of Experts as Unified Competitive Learning* (USMoE)
* *TopK Language Models*
* *OMoE: Orthogonal Mixture of Experts*
* work on representation collapse in SMoEs
* *Monet: Mixture of Monosemantic Experts for Transformers*
* *CompeteSMoE*

### 3.1 USMoE: a competition-flavored routing knob

USMoE starts from an observation that is broadly aligned with the competitive learning lens:

* **Token choice** routing (selecting experts independently for each token) can over-focus on “irrelevant” experts for certain tasks (they emphasize text embeddings / MTEB).
* **Expert choice** routing (allocating tokens to experts in a more global fashion) can discard important tokens.

They frame these as two competitive learning regimes and propose a “unified competitive learning” scheme that interpolates between them by taking a **convex combination** of token-choice and expert-choice scores.

From the sparse coding viewpoint:

* USMoE stays in the **router-centric regime**. There is still a separate scoring network that operates in a low dimensional routing space.
* Competition is defined over router scores, not over the actual energy captured by experts in the residual stream.
* There is no explicit notion of expert geometry or dictionary conditioning.

So USMoE is useful as a diagnostic: it backs up the idea that competitive behavior of experts matters for generalization, especially off the autoregressive training path. But algorithmically, it is a **routing knob**, not a rethink of what the experts are or how they interact with the residual stream.

For our purposes, it is a reminder that downstream tasks like embeddings are sensitive to subtle routing pathologies. It does not directly tell us how to design a better set of experts.

### 3.2 TopK LMs: hard sparsity baked into the architecture

TopK Language Models take a different tack. Instead of post-hoc sparse autoencoders (SAEs), they modify the transformer architecture so that certain hidden layers apply a **TopK activation**, turning the hidden state itself into the latent code of an SAE.

Conceptually:

* The read matrix (V) is the incoming projection into the hidden layer.
* A hard TopK nonlinearity enforces sparsity in the hidden activations.
* The write matrix (U) maps these sparse codes back to the residual stream.

This is almost the simplest possible way to bake sparse coding into the forward pass. It shares some motivations with our setup:

* Fine grained sparsity
* Aligning model internals with a feature basis that is interpretable and steerable
* Avoiding the ambiguity of post-hoc SAEs

However, from a competitive learning and geometry perspective, TopK LMs largely stop at “TopK exists”:

* There is no structured blocking of the dictionary into experts.
* There is no explicit competition story beyond “be in the top k activations.”
* There is no attempt to encourage approximate orthogonality, diversity, or non-redundancy among dictionary atoms, beyond whatever emerges from the task loss.

This makes TopK LMs a great **baseline** and sanity check. They show that you can make activations sparse without immediately tanking performance, and that doing so helps interpretability. They do not yet exploit the full space of ideas around energy-based competition, expert geometry, and RIP-like conditioning.

### 3.3 OMoE: orthogonalizing writes to fight redundancy

OMoE is motivated by a problem that is front and center in the sparse coding view: **expert homogeneity**. They point out that in many MoE models, expert representations end up highly similar, with some layers showing up to 99 percent similarity between experts.

Their fix is to introduce an **orthogonal expert optimizer** that, for each input, orthogonalizes the **writes** of the active experts. Roughly:

* Given the per-expert outputs, they apply a Gram–Schmidt-like procedure so that later experts contribute only the components of their output that are not already spanned by earlier experts.
* This implicitly penalizes redundant experts, since redundant directions get projected away and receive less gradient signal.

The grocery-shopping analogy is helpful here:

> You send multiple shoppers to buy groceries. When they return, you unpack them one by one. For each shopper, you keep only the items that nobody has bought yet and throw away duplicates.
> Shoppers who keep buying bananas after others already did will have most of their contribution discarded. Over time they learn to specialize in other parts of the shopping list.

In the MoE context:

* The “shopping list” is the residual stream update.
* Experts whose writes lie in already-covered directions get their contributions projected out and starve of gradient.
* This encourages diversity in what experts write back, without ever touching the read side.

From our sparse coding lens, OMoE is interesting but asymmetric:

* It operates entirely on **activations**, not on weights.
* It attacks redundancy in **writes**, whereas our focus is primarily on the **read geometry** and weight-level incoherence.
* It enforces a strict, input-dependent orthogonalization that may throw away useful signal, rather than asking for approximate orthogonality in the learned dictionary itself.

So OMoE provides evidence that explicitly encouraging diversity among experts is useful and trainable. It also suggests a complementary axis to ours: we focus on making reads well conditioned and diverse, they focus on orthogonalizing writes at run time. A more unified picture would reason jointly about both.

### 3.4 Representation collapse in SMoEs: routing geometry as a failure mode

Work on representation collapse in sparse MoEs starts from another pathology that becomes obvious once you look at routing through a geometric lens:

* Standard router architectures often map residual states into a smaller routing space and then learn centroids or prototypes in that space.
* Tokens get routed based on proximity to these centroids.
* Over training, token representations are implicitly encouraged to **cluster around combinations of expert centroids**, reducing diversity in the original feature space.

This is a form of collapse: the model learns to organize its internal representation around a small number of routing prototypes, which is not obviously aligned with the true structure of the task.

One line of work addresses this by **estimating routing scores on a low dimensional hypersphere**. Informally:

* Do the clustering and competition in a reduced space where collapse is “allowed.”
* Try to preserve flexibility and variation in the original residual stream.

Within our sparse coding framing, this is a partial fix. It accepts the basic router-centric setup and tries to avoid its worst geometric side effects by modifying the router space.

In contrast, if we route based on **read energy** (|V_i^\top x|_2^2) directly:

* The “centroids” are literally the columns of (V), not separate router embeddings.
* Tokens are encouraged to cluster around directions in the read dictionary.
* If that dictionary is well conditioned and incoherent, clustering is exactly what we want: it means different experts are truly specializing to different slices of the manifold.

So representation collapse papers are valuable in that they sharpen the diagnosis: naive routing geometries can distort the internal representation in harmful ways. Our approach tries to sidestep this by making the **dictionary itself** the geometry that routing is based on, and by shaping that dictionary to have good properties.

### 3.5 Monet: scaling expert count via product keys

Monet pursues a different axis: extremely large expert counts and mechanistic interpretability.

They combine:

* a product key style addressing scheme that lets them scale the number of experts to (262{,}144) per layer while keeping parameter growth roughly proportional to (\sqrt{\text{num experts}}), and
* a sparse dictionary learning objective embedded directly into the MoE, with the goal of learning monosemantic experts whose knowledge can be individually inspected and edited.

Motivationally this is close to our direction:

* Experts are treated as carriers of specific “knowledge slices.”
* There is a desire for mutual exclusivity of experts and explicit manipulation of domains, languages, and toxicity.

However, algorithmically the emphasis is on **indexing and scale**:

* Product keys give a clever way to address many experts efficiently.
* There is less focus on the fine geometry of the reads and writes, or on maintaining RIP-like properties of the dictionary.
* Diversity is encouraged mostly by sheer number of experts and specialization pressure, rather than by explicit incoherence constraints.

From the sparse coding perspective, Monet is a “brute force” point in the design space: crank up expert count and add some sparse learning structure, without deeply shaping the underlying geometry. It shows that expert granularity and interpretability can be improved dramatically. It leaves open how much additional benefit we can get by being more deliberate about the dictionary itself.

### 3.6 CompeteSMoE: approximating competition with a router

CompeteSMoE is probably the closest in spirit to the **energy-based competition** story.

At a high level:

* They define a competition score for each expert based on its **activation norm**. In their notation, something like (s_i = |g(z, W_{e_i})|^2), which is very similar to our notion of expert “energy.”
* Ideally, they would route by actually computing all expert activations and then selecting experts with highest scores.
* This is computationally expensive, so they train a **router** to predict the competition outcome and use that instead.

In other words:

1. Define a conceptually clean but expensive competition rule (pick experts with high activation energy).
2. Distill this rule into a cheaper scoring function.
3. Use the distilled router at scale.

Pierce’s take, which I agree with, is that this is a reasonable engineering compromise if you do not have a better handle on efficiency. From our viewpoint, a few points stand out:

* Their “energy” is defined over the **full expert output**, combining read and write. We are more interested in energy on the **read side** (V_i^\top x), for both geometric and interpretability reasons.
* They **do not shape** the geometry of expert reads or writes. There is no coherence penalty, no RIP-like constraints, no explicit attempt to prevent cheating.
* They do not fully resolve the computational question. They sidestep it by introducing a router that approximates the competition distribution.

This puts CompeteSMoE in an interesting position in the design space:

* It reinforces that competition based on activation magnitude is a useful signal.
* It hints at a “teacher router” pattern where competition is used to supervise a cheaper routing function.
* It does not yet connect that competition to a principled sparse coding geometry.

One of the open questions for our direction is whether we want a similar two-stage story (competition during training, router at inference) or whether we can make the energy-based routing itself efficient enough to be the primary mechanism.

---

This is a natural stopping point for (a) and (b). The next section can dive into our concrete architecture:

* how we parameterize (V) and (U)
* how we define and regularize energy-based competition
* how we handle efficiency and distributed training
* how this behaves in nanogpt-scale experiments

We can shape that once you share the latest details for (c).
