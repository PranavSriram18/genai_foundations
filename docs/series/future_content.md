
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

