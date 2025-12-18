* In our setup, experts **compete** to explain an input by capturing “energy” from different directions in representation space.
* The goal is for experts to learn a set of near-orthogonal “views” of the residual stream that
specialize to different regions of the input manifold.



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

