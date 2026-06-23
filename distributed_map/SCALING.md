# Scaling Analysis — Distributed MAP-DPP

This document analyses the computational complexity of `distributed_map` under
a joint scaling law that keeps the per-GPU workload constant as the number of
GPUs grows.

---

## 1. Problem Setup and Notation

| Symbol | Meaning | Default |
|--------|---------|---------|
| `N` | total candidate items | 1024 |
| `D` | input embedding dimension | 768 |
| `d` | projected (JL) dimension | 128 |
| `L` | total selected items | 32 |
| `S` | local trajectories per GPU | 64 |
| `P` | world size (number of GPUs) | 1 |
| `w` | beam width: candidates per selected item | — |
| `M` | local microset size per GPU | `N / P` |

The linear-kernel DPP score for a greedy selection of `L` items from `N`
candidates is:

```
log det K_S  =  Σ_{t=0}^{L-1}  log d²_t(i_t)
```

where `d²_t(i)` is the squared distance of item `i` to the subspace spanned
by the previously selected items, maintained as a running residual norm via
Gram–Schmidt.

---

## 2. Data Path and Matmul Shapes

### 2.1 Johnson–Lindenstrauss Projection

Raw embeddings are projected once before any selection:

```
Ẽ = E · W                    E ∈ ℝ^{N×D},  W ∈ ℝ^{D×d},  Ẽ ∈ ℝ^{N×d}
```

`W` is drawn from `𝒩(0, 1/d)` with a fixed seed, so every rank produces an
**identical** `W` without communication.

Cost: `O(N · D · d)` — a one-time preprocessing step.

### 2.2 Greedy Gram–Schmidt Loop

At each of the `L` selection steps, for each of the `S` trajectories, two
matmuls are performed.

#### Step (a) — inner-product scan

Find the kernel value between all `N` candidates and the currently selected
item `iₜ`:

```
e ∈ ℝ^N,    eᵢ = ⟨x̃ᵢ, x̃_{iₜ}⟩            (N, d) × (d,) → (N,)
```

This is equivalent to a matrix–vector product of the full projected matrix
against a single row.

Cost per step: `O(N · d)`.

#### Step (b) — history projection (Gram–Schmidt correction)

Subtract the component of `e` lying in the span of the `t` previously
computed basis vectors `c₀, …, c_{t-1}`:

```
coeffs ∈ ℝ^t,     coeffsₖ = cₖ[iₜ]          (t,) scalar loads
proj   ∈ ℝ^N,     proj = coeffs · C_{:t, :}   (t,) × (t, N) → (N,)
```

Cost at step `t`: `O(t · N)`.

Summed over all `L` steps:

```
Σ_{t=0}^{L-1} O(t · N)  =  O(L² · N / 2)  =  O(L² · N)
```

#### Total per-rank cost (current implementation, full-N sweep)

Multiplying by `S` independent trajectories:

```
JL projection   :  O(N · D · d)
Inner-product   :  O(S · L · N · d)
History proj    :  O(S · L² · N)
────────────────────────────────────────
Total           :  O(S · L² · N)        [history proj dominates for large L]
```

---

## 3. The Joint Scaling Regime: N = L·w, L = L₀·P

### 3.1 Definitions

Fix a **base selection count** `L₀` and a **beam width** `w`.  
Scale the number of GPUs `P` such that:

```
L  =  L₀ · P          (total selections grow with P)
N  =  L · w  =  L₀ · P · w    (total candidates grow with P)
```

The local microset per GPU is:

```
M  =  N / P  =  L₀ · w        (constant, independent of P)
```

Each GPU is responsible for selecting `L₀` items from its `M = L₀·w` local
candidates. The ratio `M / L₀ = w` is constant: there are always exactly `w`
candidates per local selection slot.

### 3.2 Per-rank Cost in the Current Implementation

Substituting `N = L₀·P·w` and `L = L₀·P` into the formulas from Section 2:

| Term | Formula | After substitution | Scaling in P |
|------|---------|-------------------|--------------|
| JL projection | `N·D·d` | `L₀·P·w·D·d` | **O(P)** |
| Inner-product scan | `S·L·N·d` | `S · L₀P · L₀Pw · d` | **O(P²)** |
| History projection | `S·L²·N` | `S · (L₀P)² · L₀Pw` | **O(P³)** ❌ |
| `cis` tensor (memory) | `S·L·N` fp32 | `S · L₀P · L₀Pw` | **O(P²)** ❌ |

> The current implementation sweeps all `N` items in every step (see
> `di2s` and `cis` shapes `(S, N)` and `(S, L, N)` in `kernels.py`).
> This does **not** benefit from the constant-microset invariant.

### 3.3 Local-Partition Redesign

Each GPU sweeps only its own `M = N/P = L₀·w` candidates and selects `L₀`
items locally. The global result is the union of all rank-local selections,
yielding `L₀ · P = L` total items.

Substituting `M = L₀·w` and `L_local = L₀` (both constant in `P`):

| Term | Formula (local) | After substitution | Scaling in P |
|------|----------------|-------------------|--------------|
| JL projection | `M·D·d` | `L₀·w·D·d` | **O(1)** ✓ |
| Inner-product scan | `S·L₀·M·d` | `S·L₀²·w·d` | **O(1)** ✓ |
| History projection | `S·L₀²·M` | `S·L₀³·w` | **O(1)** ✓ |
| `cis` tensor (memory) | `S·L₀·M` fp32 | `S·L₀²·w` | **O(1)** ✓ |

All per-rank costs are **constant in `P`**.

---

## 4. Derivation of the O(M) Local Complexity

We now fix `L₀`, `S`, `d` as constants and ask: how does the per-GPU cost
scale with the microset size `M`?

### Inner-product scan

At each of the `L₀` steps, for each of the `S` trajectories:

```
cost_step = M · d       (one (M, d) × (d,) product)
```

Summed over `L₀` steps and `S` trajectories:

```
C_inner = S · L₀ · M · d  =  Θ(M)      (L₀, S, d  fixed)
```

### History projection

At step `t ∈ {0, …, L₀-1}`, for each of the `S` trajectories:

```
cost_step(t) = t · M       (one (t,) × (t, M) product)
```

Summed over all steps:

```
C_hist = S · Σ_{t=0}^{L₀-1} t · M
       = S · M · L₀(L₀-1)/2
       = Θ(M)              (L₀, S  fixed)
```

### Total per-GPU complexity

```
C_total = C_inner + C_hist
        = S · L₀ · M · d  +  S · M · L₀(L₀-1)/2
        = S · M · [ L₀·d  +  L₀(L₀-1)/2 ]
        = Θ(M)
```

**Per-GPU complexity is linear in the microset size `M = N/P`.**

### Why this is optimal

Each of the `M` candidates must be read at least once per selection step to
update its residual norm `d²_t`. A single step therefore has a lower bound of
`Ω(M)` reads. Summed over `L₀` steps: `Ω(L₀ · M)` — linear in `M`. The
local-partition design matches this lower bound up to the constant `S·d`.

---

## 5. Trajectories Per GPU

`S` (local trajectories) is a fixed config parameter in `DistributedMAPConfig`.
It does not appear in `N`, `L`, or `P`. Each GPU always runs exactly `S`
independent trajectories, regardless of world size.

The Triton kernel maps one CUDA program per trajectory (`grid = (n_traj,)`),
so adding GPUs adds trajectory capacity proportionally without changing the
per-trajectory work.

---

## 6. Communication

After local selection, ranks exchange results via NCCL:

| Operation | Payload | Cost |
|-----------|---------|------|
| `all_reduce(score)` | 1 × fp32 | O(1) |
| `all_reduce(start_value)` | 1 × int64 | O(1) |
| `all_reduce(rank_value)` | 1 × int64 | O(1) |
| `broadcast(selected)` — current | `L₀` × int64 per rank | O(L₀) |
| `all_gather(selected)` — local-partition | `P · L₀` × int64 total | O(L) = O(P) |

The `all_gather` in the local-partition design grows as `O(L) = O(L₀·P)`.
This is unavoidable: the output itself has size `L`. The
**communication-to-computation ratio** is:

```
comm / compute  ≈  O(L₀·P) / O(S·L₀²·M)
               =  O(P) / O(1)
               →  ∞  as  P → ∞
```

At large `P`, the algorithm becomes **communication-bound**. The crossover
point depends on NCCL latency (`α`) and bandwidth (`β`):

```
P_crossover  ≈  S · L₀ · M · d · (1/α)      [latency-dominated regime]
```

---

## 7. Full Scaling Summary

```
                          │ Current impl    │ Local-partition design
──────────────────────────┼─────────────────┼────────────────────────
Trajectories per GPU      │ S  (constant)   │ S  (constant)
Local microset            │ N  (grows as P) │ M = L₀·w  (constant)
Per-GPU compute           │ O(P³)  ❌       │ O(M) = O(1)  ✓
Per-GPU memory (cis)      │ O(P²)  ❌       │ O(S·L₀·M) = O(1)  ✓
Local complexity in M     │ O(M³)           │ O(M)  ✓  (optimal)
Communication             │ O(L₀) = O(1)   │ O(L) = O(P)
New bottleneck at large P │ compute + OOM   │ all_gather latency
Diversity guarantee       │ global (exact)  │ intra-partition (approx)
```

### Key takeaways

1. **Constant trajectories per GPU**: `S` is fixed; adding GPUs adds trajectory
   capacity proportionally.

2. **Linear local complexity**: with `L₀`, `S`, `d` fixed, per-GPU work scales
   as `Θ(M)` — matching the read-lower-bound for the local problem.

3. **The `L²` history term is a constant prefactor here**: because `L_local =
   L₀` is fixed, the `L²·M` history cost becomes `L₀²·M`, which is `Θ(M)`.

4. **Weak scaling is perfect**: doubling GPUs doubles `L` and `N`, halves `M`
   per GPU... wait — `M = L₀·w` is **constant by construction**. There is
   nothing to halve. Every GPU always sees exactly `M` candidates.

5. **The only growing cost is the answer itself**: returning `L = L₀·P` items
   requires `O(L)` communication, which is inherent to the problem.

6. **Cross-partition diversity is approximate**: global DPP diversity requires
   Gram–Schmidt corrections across partition boundaries. Enforcing it exactly
   would require `O(S·L·d)` per-step all-reduces, costing `O(P)` per step and
   `O(L·P) = O(P²)` total — eliminating the scaling benefit.
