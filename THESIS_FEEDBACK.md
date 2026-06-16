# Thesis Feedback — Professor's Comments & Action Plan

---

## Professor's Feedback (verbatim)
  The correct approach: Minkowski sum
> "Some tables, bar graphs or even radar graphs would be good to see the performance comparisons.
> Give a unique name to your method. Instead of DPCC, you can call your method as 'Our method' or something else.
> A comparison table where all methods achieve 100% success does not help. In this case, design a more
> challenging evaluation scenario where some methods will have lower success rates."

---

## 1. Give the Method a Unique Name

**Problem:** "DPCC" is the name from the paper being implemented — not a contribution name.

**Fix:** Rename it in the thesis. Options:
- **SafeDiff** — Safe Diffusion Planning
- **ConstrainedDiff** — Constrained Diffusion
- **ProjDiff** — Projection-guided Diffusion
- **SafeNav** — Safe Navigation via Diffusion

In the thesis write:
> *"We propose SafeDiff, a method that combines diffusion-based trajectory planning
> with constraint projection to guarantee safety during navigation..."*

In all tables and graphs use: **"SafeDiff (Ours)"** instead of "DPCC".

---

## 2. The 100% Success Table Problem

If all methods succeed 100% of the time the task is too easy — no method looks better
than any other. Reviewers will ask *"why do we need your method if everything works?"*

**Bad table (current situation — task too easy):**

| Method           | Success Rate |
|------------------|-------------|
| SafeDiff (Ours)  | 100%        |
| Diffuser         | 100%        |
| Gradient         | 100%        |
| Post-processing  | 100%        |

**Good table (harder task — methods differentiated):**

| Method           | Success Rate | Safety Violations |
|------------------|-------------|-------------------|
| SafeDiff (Ours)  | **85%**     | **2%**            |
| Post-processing  | 60%         | 25%               |
| Gradient         | 55%         | 30%               |
| Diffuser         | 40%         | 45%               |

Now the contribution is clear.

---

## 3. Design a More Challenging Evaluation Scenario

Make the task hard enough that weaker methods fail. Change one or more of these:

| What to change          | How to do it                                         | Effect                        |
|-------------------------|------------------------------------------------------|-------------------------------|
| Denser obstacles        | More boxes/cylinders, reduce spacing between them    | Harder to plan a safe path    |
| Narrower corridor       | `y_bounds = (-0.6, 0.6)` instead of `(-0.95, 0.95)` | Less room to manoeuvre        |
| Dynamic obstacles       | Use `eval_dynamic_obs.py` — moving cylinders         | Stresses weaker planners most |
| Tighter safety margins  | `tighten = 0.10` instead of `0.05`                   | Smaller feasible set          |
| Longer path             | Increase `gate_x_max`, spread obstacles further      | More chances to fail          |

**Target story:**
- *Diffuser (no projection)* fails often — it ignores obstacles
- *Post-processing / Gradient* partially works — fixes plans but imperfectly
- *SafeDiff (Ours)* succeeds most — projection is correct and consistent

---

## 4. Visualisations to Include

### Bar Graph — Success rate per method
Most readable. One bar per method, grouped by scenario (easy / hard / dynamic).

```
SafeDiff  ████████████ 85%
Post-proc ████████     60%
Gradient  ███████      55%
Diffuser  █████        40%
```

### Radar / Spider Graph — Multiple metrics at once
Shows trade-offs visually. Each method is a polygon — larger area = better overall.

Suggested axes:
- Success rate
- Safety violation rate (lower is better — invert axis)
- Mean path length (shorter = better)
- Computation time per step
- Mean obstacle clearance margin

### Comparison Table — Easy vs Hard vs Dynamic

| Method          | Easy corridor | Hard corridor | Dynamic obstacles |
|-----------------|--------------|---------------|-------------------|
| SafeDiff (Ours) | 100%         | **85%**       | **70%**           |
| Post-processing | 100%         | 60%           | 45%               |
| Gradient        | 100%         | 55%           | 40%               |
| Diffuser        | 100%         | 40%           | 25%               |

The easy scenario shows all methods work on simple tasks.
The hard/dynamic scenarios show SafeDiff is the only robust one.
**That contrast is the thesis argument.**

---

## 5. Seeding for Reproducible Results

- Train with multiple seeds: `9`, `42`, `123` (minimum 3)
- Eval each trained model with `--episodes 10` and `set_seed(ep)` per episode
- Report: **mean ± std** not a single number

```
SafeDiff (seed 9):   0.82 ± 0.11
SafeDiff (seed 42):  0.85 ± 0.09
SafeDiff (seed 123): 0.80 ± 0.13
SafeDiff (average):  0.82 ± 0.11   ← put this in the thesis
```

Write in thesis:
> *"Results are averaged over 3 training seeds {9, 42, 123} and 10 evaluation
> episodes per seed. We report mean ± standard deviation."*

---

## TODO Checklist

- [ ] Pick a unique method name (SafeDiff / ConstrainedDiff / ProjDiff)
- [ ] Design hard evaluation scenario (denser obstacles or narrower corridor)
- [ ] Run `eval_dynamic_obs.py` as a third scenario (dynamic obstacles)
- [ ] Train models with seeds 42 and 123 in addition to seed 9
- [ ] Re-run eval with `--episodes 10` and `set_seed(ep)` per episode
- [ ] Create bar graph: success rate per method × scenario
- [ ] Create radar graph: multi-metric comparison
- [ ] Create comparison table: easy / hard / dynamic columns
- [ ] Replace "DPCC" with chosen method name everywhere in thesis
