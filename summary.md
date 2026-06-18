# CADEMAS-ML — Technical Summary

## 1. Overview

**CADEMAS-ML** is a web-based decision support system that implements a cooperative, context-aware prioritization pipeline. For each case \(i\) in a dataset, the system computes:

1. a **global machine-learning risk** $R_i \in [0,1]$, obtained from an ensemble of H2O MOJO models;
2. a **context alignment score** \(C_i \in [0,1]\), obtained from expert-defined fuzzy rules over contextual features;
3. a **prioritization score** \(P_i \in [0,1]\), defined as a convex combination of \(R_i\) and \(C_i\) controlled by a user parameter \(\lambda\).

The design follows the CADEMAS framework (Cooperative Automated Decision-Making Systems): predictive models supply data-driven evidence, while a declarative context layer encodes organizational policy, domain expertise, and situational priorities. Both components remain explicit, auditable, and independently inspectable in the application interface.

---

## 2. Inputs and preprocessing

The system requires four mandatory inputs:

| Input | Role |
|---|---|
| Model performance configuration (JSON) | Defines per-model performance metrics used to compute ensemble weights |
| Context configuration (JSON) | Defines fuzzy rules, derived compositions, and optional final logic |
| MOJO models (`.zip`) | H2O-exported models for batch inference |
| Case dataset (CSV) | Cases to be evaluated |

**Case identification.** If the dataset contains a `Case_ID` or `ID` column, its values are adopted as the canonical identifier `CaseID`. Otherwise, consecutive integers \(1, \ldots, n\) are assigned. Identifier columns are excluded from model inference.

**CSV parsing.** Delimiters (comma, semicolon, tab) are auto-detected by selecting the parse that yields the largest number of columns.

**Model heterogeneity.** Uploaded models may have been trained on different feature subsets. H2O handles feature alignment at prediction time; the application passes the full feature matrix (excluding identifiers) to each MOJO model.

---

## 3. Cooperative machine-learning risk aggregation

Let \(\mathcal{M} = \{1, \ldots, M\}\) denote the set of uploaded models whose filenames match entries in the performance configuration. For a user-selected metric \(m\) (e.g. AUC, F1), let \(s_j\) be the performance value of model \(j\) on metric \(m\). Normalized weights are:

\[
w_j = \frac{s_j}{\sum_{k \in \mathcal{M}} s_k}, \quad j \in \mathcal{M}
\]

For each case \(i\), model \(j\) produces a positive-class probability \(\hat{p}_{ij} \in [0,1]\) via MOJO inference. The **global ML risk** is the weighted ensemble:

\[
R_i = \sum_{j \in \mathcal{M}} w_j \, \hat{p}_{ij}
\]

Properties:

- Weights are **performance-driven**, not learned at runtime; they reflect prior validation results supplied by the analyst.
- All models must agree on the predicted positive label across the batch; otherwise the pipeline aborts with an error.
- Individual model probabilities are retained for per-model inspection in the **Models** tab.

---

## 4. Fuzzy context evaluation

The context layer maps raw case features to a scalar alignment score \(C_i\) through a structured fuzzy inference pipeline. This section describes the formal model implemented in `fuzzy_context.py`.

### 4.1. Design rationale

Classical crisp rules (e.g. *if salary < X then priority = high*) discard graded evidence and are brittle at decision boundaries. Fuzzy set theory represents partial satisfaction of conditions through **membership degrees** \(\mu \in [0,1]\), enabling:

- smooth transitions between policy regimes;
- compositional reasoning via t-norms and s-norms;
- full traceability of how each feature contributes to the final score.

In CADEMAS-ML, context is not a post-hoc explanation layer: it is a **first-class decision ingredient** combined with ML risk through \(\lambda\).

### 4.2. Configuration schema

A context configuration is a JSON document with the following structure:

```json
{
  "context_name": "...",
  "description": "...",
  "rules": [ ... ],
  "derived_rules": [ ... ],
  "logic": { ... }
}
```

- **`rules`**: atomic fuzzy rules, each mapping one feature to a membership degree (or categorical label).
- **`derived_rules`**: composite rules that aggregate outputs of other rules via fuzzy operators.
- **`logic`**: optional tree-structured expression for the final context score (used when no UI aggregation override applies).

### 4.3. Atomic rules

Each atomic rule \(r\) is defined by:

| Field | Description |
|---|---|
| `name` | Unique identifier for the rule |
| `feature` | Column name in the case dataset |
| `type` | Membership function family |
| `params` | Parameters of the membership function |
| `when` (optional) | Conditional activation mask |

For case \(i\), let \(x_{ir}\) denote the value of feature \(r\) for that case. The **atomic membership** is:

\[
\mu_r(i) = \begin{cases}
f_r(x_{ir};\,\theta_r) & \text{if } \mathrm{when}_r(i) = \mathrm{true} \\
0 & \text{otherwise}
\end{cases}
\]

where \(f_r\) is selected by `type` and \(\theta_r\) is `params`.

#### 4.3.1. Numeric membership functions

All numeric functions are piecewise linear and map to \([0,1]\).

**Triangular** (`triangular`, parameters \(a < b < c\)):

\[
\mu(x) = \max\!\left(0,\; \min\!\left(\frac{x-a}{b-a},\; \frac{c-x}{c-b}\right)\right)
\]

**Trapezoidal** (`trapezoidal`, parameters \(a < b \leq c < d\)):

\[
\mu(x) = \max\!\left(0,\; \min\!\left(\frac{x-a}{b-a},\; 1,\; \frac{d-x}{d-c}\right)\right)
\]

**Linear increasing** (`linear_increasing`, parameters \(a < b\)):

\[
\mu(x) = \max\!\left(0,\; \min\!\left(\frac{x-a}{b-a},\; 1\right)\right)
\]

**Linear decreasing** (`linear_decreasing`, parameters \(a < b\)):

\[
\mu(x) = \max\!\left(0,\; \min\!\left(\frac{b-x}{b-a},\; 1\right)\right)
\]

#### 4.3.2. Categorical rules

**Direct mapping** (`categorical_map`): a dictionary \(\{(c_k, v_k)\}\) assigns a fixed score \(v_k \in [0,1]\) to each category \(c_k\). Unmapped categories receive \(\mu = 0\).

**Set assignment** (`categorical_set`): categories are grouped into named sets \(S_1, \ldots, S_K\). The rule output is the set label (not a numeric score). Numeric scores are assigned later, in derived rules, via an explicit `map` from set name to value.

#### 4.3.3. Conditional activation (`when`)

Rules may be restricted to a subpopulation. Supported conditions:

- `{"feature": "F", "equals": v}` — active when \(x_{iF} = v\);
- `{"feature": "F", "in": [v_1, \ldots, v_k]}` — active when \(x_{iF} \in \{v_1, \ldots, v_k\}\).

This mechanism supports department-specific salary bands, role-specific criteria, and other context-dependent policies without duplicating feature columns.

### 4.4. Derived rules

A derived rule \(d\) combines the outputs of one or more input rules \(\{r_1, \ldots, r_k\}\) through an aggregation operator \(\oplus_d\):

\[
\mu_d(i) = \oplus_d\bigl(\mu_{r_1}(i), \ldots, \mu_{r_k}(i)\bigr)
\]

When an input references a `categorical_set` rule, a per-input `map` translates set labels to numeric values before aggregation.

Supported operators (Gödel / product semantics):

| Operator | Symbol | Definition |
|---|---|---|
| `AND`, `MIN`, `MINIMUM` | t-norm (Gödel) | \(\min_j \mu_{r_j}(i)\) |
| `OR`, `MAX` | s-norm (Gödel) | \(\max_j \mu_{r_j}(i)\) |
| `PRODUCT`, `PROD` | t-norm (product) | \(\prod_j \mu_{r_j}(i)\) |
| `AVERAGE`, `MEAN` | — | \(\frac{1}{k}\sum_j \mu_{r_j}(i)\) |

Derived rules enable hierarchical policy structures. For example, *hard to replace* may be defined as the disjunction of high job level and critical role category; *salary cost acceptable* as the disjunction of department-specific salary memberships.

### 4.5. Final context score \(C_i\)

The application supports two aggregation paths:

#### Path A — UI-selected operator (default at runtime)

The user selects one of three operators in the sidebar:

| UI option | Computation over all numeric membership columns |
|---|---|
| `average` | \(C_i = \mathrm{mean}_r\, \mu_r(i)\) |
| `minimum (strict)` | \(C_i = \min_r\, \mu_r(i)\) |
| `product` | \(C_i = \prod_r\, \mu_r(i)\) |

The aggregation is applied to every numeric column in the audit matrix (atomic and derived memberships). The result is clipped to \([0,1]\).

#### Path B — Declarative logic tree (schema-level)

If no UI operator is applied, the `logic` field defines a recursive expression tree. Leaf nodes reference rules by name; internal nodes apply the operators from Section 4.4 row-wise across cases. Example:

```json
"logic": {
  "op": "AND",
  "inputs": [
    { "rule": "hard_to_replace" },
    { "rule": "tenure_relevant" },
    { "rule": "performance_at_least_medium" }
  ]
}
```

This yields \(C_i = \min\bigl(\mu_{\text{hard\_to\_replace}}(i),\, \mu_{\text{tenure\_relevant}}(i),\, \mu_{\text{performance\_at\_least\_medium}}(i)\bigr)\).

> **Note.** At runtime, the sidebar aggregation operator takes precedence over the `logic` tree. The `logic` field remains part of the configuration schema for specification, portability, and fallback evaluation in the inference engine.

### 4.6. Auditability

Every computed membership — atomic and derived — is stored in a per-case audit matrix (`fuzzy_details`). Column names follow the convention `mu_<rule_name>`. Together with membership function plots (numeric rules) and tabular membership values, this provides full transparency of the fuzzy inference chain for each case.

### 4.7. Illustrative example

In the *Economic Crisis* context (`example_attrition/context/context_crisis.json`), retention priority under crisis conditions is encoded through rules such as:

- relevant tenure (triangular over `years_at_company`);
- minimum performance (linear increasing over `performance_rating`);
- role criticality (`categorical_set` over `job_role`);
- department-conditioned salary acceptability (conditional triangular rules);
- commitment proxies (involvement and satisfaction memberships).

Derived rules compose these into higher-level constructs (`hard_to_replace`, `commitment_medium_or_high`, `salary_cost_ok`). The analyst can then select a global aggregation operator or rely on the declarative `logic` tree to obtain \(C_i\).

---

## 5. Decision integration

The **prioritization score** combines ML risk and context alignment through a convex combination parameterized by \(\lambda \in [0,1]\):

\[
P_i = \lambda \, R_i + (1 - \lambda) \, C_i
\]

Interpretation:

| \(\lambda\) | Behaviour |
|---|---|
| \(1\) | Purely ML-driven prioritization |
| \(0\) | Purely context-driven prioritization |
| \((0,1)\) | Explicit trade-off between predictive risk and policy alignment |

Decomposition for inspection:

\[
\text{ML contribution}_i = \lambda \, R_i, \qquad \text{Context contribution}_i = (1-\lambda) \, C_i
\]

The **Robustness** tab performs a \(\lambda\)-sensitivity analysis: for a grid \(\lambda \in \{0, \tfrac{1}{k}, \ldots, 1\}\), case rankings are recomputed and visualized (bump chart) to assess stability of prioritization under varying decision policies.

---

## 6. Computational pipeline

The end-to-end flow executed on **Run analysis** is:

```
CSV cases
   │
   ├─► H2O MOJO inference (per model) ──► weighted sum ──► R_i
   │
   └─► Fuzzy context engine ──► C_i
              │
              ▼
        P_i = λ·R_i + (1−λ)·C_i
              │
              ▼
   Overview / Models / Context / Robustness views
```

All intermediate artifacts (per-model probabilities, membership degrees, weights, \(\lambda\)) are retained in session state for interactive exploration without re-running inference.

---

## 7. References

1. Novoa-Hernández, P., Pelta, D. A., Godz, M., Verdegay, J. L., & Buendia-Carrillo, D. (2026). CADEMAS – A Framework for Cooperative Automated Decision-Making Systems. In *2026 IEEE Conference on Artificial Intelligence (CAI)* (pp. 756–761). IEEE. https://doi.org/10.1109/cai68641.2026.11536392

2. Godz, M., Novoa-Hernández, P., & Pelta, D. A. (2026). A Cooperative and Context-Aware Approach for Employee Attrition Prevention. In *Communications in Computer and Information Science* (pp. 203–217). Springer Nature Switzerland. https://doi.org/10.1007/978-3-032-29000-7_15
