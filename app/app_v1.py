from traceback import print_last

import streamlit as st
import h2o
import pandas as pd
import numpy as np
import json
import os
import tempfile
import altair as alt
import streamlit.components.v1 as components
from h2o.model import ModelBase

st.set_page_config(page_title="CADEMAS-ML – Cooperative and Context-Aware Decision Support", layout="wide")

custom_css = """
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:11pt;
    align-items:center;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


def render_animated_header():
    """
    Renderiza el diagrama de flujo animado usando un iframe aislado
    para garantizar que los estilos y animaciones funcionen en cualquier navegador.
    """
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@100..900&display=swap');
</style>
        <style>
            body { margin: 0; padding: 0; background-color: transparent; font-family: 'Geist Mono'; overflow: hidden; }
            .container {
                width: 100%;
                height: 180px;
                background: linear-gradient(90deg, #0e1117 0%, #1a1c24 100%);
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            svg { width: 100%; height: 100%; max-width: 900px; }

            /* Estilos estáticos */
            .node-rect { fill: #1f2937; stroke: #374151; stroke-width: 2px; rx: 6px; }
            .text-title { fill: #f3f4f6; font-size: 12px; font-weight: semibold; font-family: 'Geist Mono', monospace; pointer-events: none; }
            .text-sub { fill: #9ca3af; font-size: 10px; font-family: 'Geist Mono', monospace; pointer-events: none; }

            /* Colores de los nodos */
            .stroke-ml { stroke: #3b82f6; }
            .stroke-fuzzy { stroke: #f97316; }
            .stroke-hybrid { stroke: #8b5cf6; }

            /* Caminos */
            .path-line { fill: none; stroke: #4b5563; stroke-width: 2px; opacity: 0.3; }

            /* Partículas brillantes */
            .dot { fill: white; filter: drop-shadow(0 0 4px rgba(255,255,255,0.8)); }
            .dot-ml { fill: #60a5fa; }
            .dot-fuzzy { fill: #fb923c; }
            .dot-hybrid { fill: #a78bfa; }
        </style>
    </head>
    <body>
        <div class="container">
            <svg viewBox="0 0 800 160" preserveAspectRatio="xMidYMid meet">
                <defs>
                    <path id="p1" d="M 90 50 L 240 50" />
                    <path id="p2" d="M 90 110 L 240 110" />
                    <path id="p3" d="M 340 50 L 460 80" />
                    <path id="p4" d="M 340 110 L 460 80" />
                    <path id="p5" d="M 560 80 L 670 80" />
                </defs>

                <path d="M 90 50 L 240 50" class="path-line" />
                <path d="M 90 110 L 240 110" class="path-line" />
                <path d="M 340 50 L 460 80" class="path-line" />
                <path d="M 340 110 L 460 80" class="path-line" />
                <path d="M 560 80 L 670 80" class="path-line" />

                <g transform="translate(10, 30)">
                    <rect width="80" height="40" class="node-rect" />
                    <text x="40" y="20" text-anchor="middle" dominant-baseline="middle" class="text-title">DATA</text>
                    <text x="40" y="32" text-anchor="middle" class="text-sub">CSV</text>
                </g>
                <g transform="translate(10, 90)">
                    <rect width="80" height="40" class="node-rect" />
                    <text x="40" y="20" text-anchor="middle" dominant-baseline="middle" class="text-title">RULES</text>
                    <text x="40" y="32" text-anchor="middle" class="text-sub">JSON</text>
                </g>

                <g transform="translate(240, 30)">
                    <rect width="100" height="40" class="node-rect stroke-ml" />
                    <text x="50" y="20" text-anchor="middle" dominant-baseline="middle" class="text-title">ML ENSEMBLE</text>
                    <text x="50" y="32" text-anchor="middle" class="text-sub">Riesgo (Ri)</text>
                </g>
                <g transform="translate(240, 90)">
                    <rect width="100" height="40" class="node-rect stroke-fuzzy" />
                    <text x="50" y="20" text-anchor="middle" dominant-baseline="middle" class="text-title">FUZZY LOGIC</text>
                    <text x="50" y="32" text-anchor="middle" class="text-sub">Contexto (Ci)</text>
                </g>

                <g transform="translate(460, 60)">
                    <rect width="100" height="40" class="node-rect stroke-hybrid" />
                    <text x="50" y="20" text-anchor="middle" dominant-baseline="middle" class="text-title">HYBRID CORE</text>
                    <text x="50" y="32" text-anchor="middle" class="text-sub">λ•Ri + (1-λ)•Ci</text>
                </g>

                <g transform="translate(670, 60)">
                    <rect width="100" height="40" class="node-rect" />
                    <text x="50" y="20" text-anchor="middle" dominant-baseline="middle" class="text-title">DASHBOARD</text>
                    <text x="50" y="32" text-anchor="middle" class="text-sub">Decisión</text>
                </g>

                <circle r="4" class="dot dot-ml">
                    <animateMotion dur="2s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear">
                        <mpath href="#p1"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2s" repeatCount="indefinite" />
                </circle>

                <circle r="4" class="dot dot-fuzzy">
                    <animateMotion dur="2.5s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear">
                        <mpath href="#p2"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2.5s" repeatCount="indefinite" />
                </circle>

                <circle r="4" class="dot dot-ml">
                    <animateMotion dur="2s" begin="1s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear">
                        <mpath href="#p3"/>
                    </animateMotion>
                     <animate attributeName="opacity" values="0;1;1;0" dur="2s" begin="1s" repeatCount="indefinite" />
                </circle>

                <circle r="4" class="dot dot-fuzzy">
                    <animateMotion dur="2.5s" begin="1.2s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear">
                        <mpath href="#p4"/>
                    </animateMotion>
                     <animate attributeName="opacity" values="0;1;1;0" dur="2.5s" begin="1.2s" repeatCount="indefinite" />
                </circle>

                 <circle r="5" class="dot dot-hybrid">
                    <animateMotion dur="3s" begin="0.5s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear">
                        <mpath href="#p5"/>
                    </animateMotion>
                     <animate attributeName="opacity" values="0;1;1;0" dur="3s" begin="0.5s" repeatCount="indefinite" />
                </circle>

            </svg>
        </div>
    </body>
    </html>
    """
    # Renderizamos el componente HTML con una altura fija para evitar scrollbars
    components.html(html_code, height=190, scrolling=False)


# --- 1. Inicialización ---
@st.cache_resource
def init_h2o():
    try:
        h2o.init()
        return True
    except:
        return False


if not init_h2o(): st.stop()


# --- 2. Funciones Lógicas ---
def save_temp_file(uploaded_file):
    try:
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return path
    except:
        return None


# --- MOTOR DIFUSO EXTENDIDO ---

def get_membership(x, m_type, params):
    """Router that selects the correct membership function."""
    if m_type == 'triangular':
        return triangular(x, *params)
    elif m_type == 'trapezoidal':
        return trapezoidal(x, *params)
    elif m_type == 'linear_increasing':
        return linear_increasing(x, *params)
    elif m_type == 'linear_decreasing':
        return linear_decreasing(x, *params)
    else:
        return np.zeros_like(x, dtype=float)

def get_categorical_membership(series: pd.Series, m_type: str, params):
    """Categorical membership for rule types:
    - categorical_map: params is a dict {category: score}
    - categorical_set: params is a dict of sets, and scores are assigned via a map in derived rules
    """
    if m_type == "categorical_map":
        mapping = params if isinstance(params, dict) else {}
        return series.map(mapping).fillna(0.0).astype(float).values

    if m_type == "categorical_set":
        # categorical_set itself returns the set name (string) per row, not a numeric score.
        # We encode as object array; derived rules can map set->score.
        groups = params if isinstance(params, dict) else {}
        out = np.array([None] * len(series), dtype=object)
        for set_name, categories in groups.items():
            if not isinstance(categories, list):
                continue
            mask = series.isin(categories)
            out[mask.values] = set_name
        # Unmatched categories remain None
        return out

    return np.zeros(len(series), dtype=float)

def apply_when_mask(df: pd.DataFrame, when: dict) -> np.ndarray:
    """Build a boolean mask for a simple conditional rule application.

    Supported forms:
      when = {"feature": "Department", "equals": "Sales"}
    """
    if not when or not isinstance(when, dict):
        return np.ones(len(df), dtype=bool)

    feat = when.get("feature")
    if feat is None or feat not in df.columns:
        return np.zeros(len(df), dtype=bool)

    if "equals" in when:
        return (df[feat] == when["equals"]).values

    if "in" in when and isinstance(when["in"], list):
        return df[feat].isin(when["in"]).values

    return np.ones(len(df), dtype=bool)

def triangular(x, a, b, c):
    """Pico en b."""
    if a == b or b == c: return np.zeros_like(x)
    term1 = (x - a) / (b - a)
    term2 = (c - x) / (c - b)
    return np.maximum(0, np.minimum(term1, term2))

def trapezoidal(x, a, b, c, d):
    """Plano (1) entre b y c."""
    # max(min( (x-a)/(b-a), 1, (d-x)/(d-c) ), 0)
    term1 = (x - a) / (b - a) if b > a else np.ones_like(x) # Protección div/0
    term3 = (d - x) / (d - c) if d > c else np.ones_like(x)
    return np.maximum(0, np.minimum(np.minimum(term1, 1), term3))

def linear_increasing(x, a, b):
    """Gamma (Sube): 0 hasta a, sube, 1 desde b en adelante."""
    if a == b: return (x >= a).astype(float)
    y = (x - a) / (b - a)
    return np.maximum(0, np.minimum(y, 1))

def linear_decreasing(x, a, b):
    """L-Shape (Baja): 1 hasta a, baja, 0 desde b en adelante."""
    if a == b: return (x <= a).astype(float)
    y = (b - x) / (b - a)
    return np.maximum(0, np.minimum(y, 1))


def _agg_values(op: str, arr: np.ndarray) -> float:
    op = (op or "AVERAGE").upper()
    if arr.size == 0:
        return 0.0
    if op in ("AVERAGE", "MEAN"):
        return float(np.mean(arr))
    if op in ("MIN", "MINIMUM"):
        return float(np.min(arr))
    if op in ("MAX",):
        return float(np.max(arr))
    if op in ("PRODUCT", "PROD"):
        return float(np.prod(arr))
    if op in ("AND",):
        # Fuzzy AND (Gödel t-norm)
        return float(np.min(arr))
    if op in ("OR",):
        # Fuzzy OR (Gödel s-norm)
        return float(np.max(arr))
    return float(np.mean(arr))

def _eval_logic_node(node, mu_dict: dict, n_rows: int) -> np.ndarray:
    """Evaluate a logic node into a vector of size n_rows.

    Node forms supported:
      - {"rule": "rule_name"}
      - {"op": "AND"|"OR"|"AVERAGE"|"PRODUCT"|"MIN"|"MAX", "inputs": [ ... ]}

    Returns a numeric array in [0,1] (best effort).
    """
    if node is None:
        return np.zeros(n_rows, dtype=float)

    # Direct reference to a rule
    if isinstance(node, dict) and "rule" in node:
        key = node["rule"]
        val = mu_dict.get(key)
        if val is None:
            return np.zeros(n_rows, dtype=float)
        # categorical_set returns object array; cannot be used directly without mapping
        if val.dtype == object:
            return np.zeros(n_rows, dtype=float)
        return val.astype(float)

    # Composite node
    if isinstance(node, dict) and "op" in node and "inputs" in node:
        op = (node.get("op") or "AVERAGE").upper()
        inputs = node.get("inputs") or []
        if not isinstance(inputs, list) or len(inputs) == 0:
            return np.zeros(n_rows, dtype=float)

        mats = []
        for child in inputs:
            mats.append(_eval_logic_node(child, mu_dict, n_rows))
        mat = np.vstack(mats)  # shape: (k, n_rows)

        if op in ("AVERAGE", "MEAN"):
            return np.mean(mat, axis=0)
        if op in ("MIN", "MINIMUM", "AND"):
            return np.min(mat, axis=0)
        if op in ("MAX", "OR"):
            return np.max(mat, axis=0)
        if op in ("PRODUCT", "PROD"):
            return np.prod(mat, axis=0)

        return np.mean(mat, axis=0)

    return np.zeros(n_rows, dtype=float)

def calculate_context_score(df: pd.DataFrame, context_config: dict, aggregation: str):
    """Compute context alignment score (Ci) supporting:

    - context_config['rules'] : base membership rules
    - context_config['derived_rules'] : composition rules built from base rules
    - context_config['logic'] : final logic expression for Ci

    Backward compatible with the old schema (rules only).

    Returns:
      scores: pd.Series
      fuzzy_details: pd.DataFrame (all computed mu_* columns + derived)
    """

    rules = context_config.get("rules", []) if isinstance(context_config, dict) else []
    derived_rules = context_config.get("derived_rules", []) if isinstance(context_config, dict) else []
    logic = context_config.get("logic") if isinstance(context_config, dict) else None

    n = len(df)

    # Store all rule outputs here by name
    mu = {}

    # Details dataframe for audit
    fuzzy_scores = pd.DataFrame(index=df.index)

    # --- 1) Base rules ---
    for i, rule in enumerate(rules):
        col = rule.get('feature')
        m_type = rule.get('type')
        params = rule.get('params')
        rule_name = rule.get('name') or f"rule_{i}_{col}"  # alias
        when = rule.get("when")

        # Default outputs
        out = np.zeros(n, dtype=float)

        # Conditional mask
        mask = apply_when_mask(df, when)

        # Numeric rule
        if col in df.columns and m_type in ("triangular", "trapezoidal", "linear_increasing", "linear_decreasing"):
            x = df[col].astype(float).values
            tmp = get_membership(x, m_type, params)
            out[mask] = tmp[mask]
            mu[rule_name] = out
            fuzzy_scores[f"mu_{rule_name}"] = out
            continue

        # Categorical rule
        if col in df.columns and m_type in ("categorical_map", "categorical_set"):
            series = df[col].astype(str)
            tmp = get_categorical_membership(series, m_type, params)
            # categorical_set returns object array (set names)
            if isinstance(tmp, np.ndarray) and tmp.dtype == object:
                # Apply mask by setting non-matching rows to None
                tmp2 = tmp.copy()
                tmp2[~mask] = None
                mu[rule_name] = tmp2
                fuzzy_scores[f"mu_{rule_name}"] = tmp2
            else:
                out[mask] = tmp[mask]
                mu[rule_name] = out
                fuzzy_scores[f"mu_{rule_name}"] = out
            continue

        # Missing feature or unsupported type
        mu[rule_name] = out
        fuzzy_scores[f"mu_{rule_name}"] = out

    # --- 2) Derived rules ---
    for d in derived_rules:
        d_name = d.get("name")
        op = (d.get("op") or "AVERAGE").upper()
        inputs = d.get("inputs") or []
        if not d_name or not isinstance(inputs, list) or len(inputs) == 0:
            continue

        # Evaluate row-wise
        out = np.zeros(n, dtype=float)

        for row_idx in range(n):
            vals = []
            for inp in inputs:
                if not isinstance(inp, dict) or "rule" not in inp:
                    continue
                key = inp["rule"]
                base_val = mu.get(key)
                if base_val is None:
                    continue

                # categorical_set mapping support: {"rule": "jobrole_critical", "map": {"critical": 1.0, ...}}
                if isinstance(base_val, np.ndarray) and base_val.dtype == object:
                    mapper = inp.get("map") if isinstance(inp.get("map"), dict) else {}
                    set_name = base_val[row_idx]
                    vals.append(float(mapper.get(set_name, 0.0)))
                else:
                    vals.append(float(base_val[row_idx]))

            if len(vals) == 0:
                out[row_idx] = 0.0
            else:
                out[row_idx] = _agg_values(op, np.array(vals, dtype=float))

        mu[d_name] = out
        fuzzy_scores[f"mu_{d_name}"] = out

    # --- 3) Final context score ---
    if logic and isinstance(logic, dict):
        # If logic is present, evaluate it
        scores_arr = _eval_logic_node(logic, mu, n)
        scores = pd.Series(scores_arr, index=df.index)
    else:
        # Backward compatible: aggregate all base mu_* columns
        if aggregation == "average":
            scores = fuzzy_scores.select_dtypes(include=[np.number]).mean(axis=1)
        elif aggregation == "minimum (strict)":
            scores = fuzzy_scores.select_dtypes(include=[np.number]).min(axis=1)
        elif aggregation == "product":
            scores = fuzzy_scores.select_dtypes(include=[np.number]).prod(axis=1)
        else:
            scores = fuzzy_scores.select_dtypes(include=[np.number]).mean(axis=1)

    # Clip to [0,1] for safety
    scores = scores.clip(0.0, 1.0)

    return scores, fuzzy_scores


# --- 3. SIDEBAR ---
with st.sidebar:
    # st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=50)
    st.title("Configuration")

    with st.expander("1. Files and Data", expanded=True):
        json_ml = st.file_uploader("Model Configuration (JSON)", type=['json'])
        json_context = st.file_uploader("Context Configuration (JSON)", type=['json'])
        model_files = st.file_uploader("MOJO Models (.zip)", type=['zip'], accept_multiple_files=True)
        data_file = st.file_uploader("Dataset (.csv)", type=['csv'])
        # ID AUTOGENERADO: No se selecciona columna, se genera "CaseID" automáticamente
        st.info(
            "Each case is automatically assigned a unique identifier named 'CaseID'."
        )
        st.session_state.record_id_col = "CaseID"


    with st.expander("2. ML Parameters", expanded=False):
        selected_metric = None
        if json_ml:
            feature_config = json.load(json_ml)
            first_key = list(feature_config.keys())[0]
            metrics = list(feature_config[first_key].get("performance", {}).keys())
            selected_metric = st.selectbox("Metric used for model weighting (w)", metrics)

    with st.expander("3. Context Parameters", expanded=False):
        aggregation_method = st.selectbox("Context aggregation operator", ["average", "minimum (strict)", "product"])

    run_calc = st.button("Run analysis", type="primary", width='stretch')

    st.markdown("## Decision Adjustment")
    lambda_val = st.slider("Lambda (weight)", 0.0, 1.0, 0.5, 0.01)
    st.caption(f"Risk contribution: {lambda_val:.0%} | Context contribution: {1 - lambda_val:.0%}")


# --- 4. ESTADO ---
if 'base_results' not in st.session_state: st.session_state.base_results = None
if 'fuzzy_details' not in st.session_state: st.session_state.fuzzy_details = None
if 'ml_details' not in st.session_state: st.session_state.ml_details = None
if 'context_config' not in st.session_state: st.session_state.context_config = None  # Guardamos config para graficar
if 'master_data' not in st.session_state: st.session_state.master_data = None  # Guardamos raw data para histogramas
if 'p_label' not in st.session_state: st.session_state.p_label = None


# --- 5. EJECUCIÓN ---
if run_calc:
    if feature_config and model_files and data_file and selected_metric and json_context:
        with st.spinner(text="Processing...", show_time=True):
            try:
                # A. Carga
                data_file.seek(0)
                master_df = pd.read_csv(data_file)
                # AUTOGENERAR ID
                master_df = master_df.copy()
                master_df.insert(0, "CaseID", np.arange(1, len(master_df) + 1))
                st.session_state.master_data = master_df  # Guardar para gráficos
                context_config = json.load(json_context)
                st.session_state.context_config = context_config  # Guardar config

                # B. Pesos ML
                valid_models = [m.name for m in model_files if m.name in feature_config]


                metrics_vals = {m: feature_config[m]["performance"].get(selected_metric, 0) for m in valid_models}



                total = sum(metrics_vals.values())
                weights = {m: (val / total if total > 0 else 1 / len(valid_models)) for m, val in metrics_vals.items()}

                st.session_state.ml_details = {"weights": weights, "metric": selected_metric}

                # C. H2O Loop
                risk_accum = np.zeros(len(master_df))
                temp_results = master_df.copy()

                prog_bar = st.progress(0)
                for i, m_file in enumerate(model_files):
                    if m_file.name not in weights: continue
                    path = save_temp_file(m_file)
                    try:
                        mojo = h2o.import_mojo(path)
                        #output = mojo._model_json['output']
                        # Columnas originales usadas por el modelo
                        #input_cols = list(output['names'])
                        # Eliminar la variable objetivo si aparece
                        #response = output.get('response_column')
                        #if response in input_cols:
                        #    input_cols.remove(response)
                        # Subconjunto seguro del master dataset
                        hf = h2o.H2OFrame(master_df)
                        preds = mojo.predict(hf).as_data_frame()
                        print(f"Columns: {preds.columns}")
                        p_col = 'p1' if 'p1' in preds.columns else preds.columns[-1]
                        current_p_label = preds.columns[-1]
                        st.session_state.p_label = current_p_label if st.session_state.p_label is None else st.session_state.p_label
                        if st.session_state.p_label != current_p_label:
                            st.error(f"Error: Predicted label is not the same across models.")
                            st.stop()
                        vals = preds[p_col].values
                        risk_accum += vals * weights[m_file.name]
                        temp_results[f"{m_file.name.split('.')[0]}_prob"] = vals
                    finally:
                        if os.path.exists(path): os.remove(path)
                        h2o.remove(hf)
                    prog_bar.progress((i + 1) / len(model_files))

                temp_results["Ri_Global_Risk"] = risk_accum

                # D. Contexto
                ci_scores, fuzzy_df = calculate_context_score(master_df, context_config, aggregation_method)
                temp_results["Ci_Context_Score"] = ci_scores

                st.session_state.base_results = temp_results
                st.session_state.fuzzy_details = fuzzy_df

                st.success("Computation completed successfully.")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("Required input files are missing.")

# --- 6. VISUALIZACIÓN ---

if st.session_state.base_results is not None:
    df = st.session_state.base_results.copy()
    df["Final_Score"] = (lambda_val * df["Ri_Global_Risk"]) + ((1 - lambda_val) * df["Ci_Context_Score"])

    df["ML_Contribution"] = lambda_val * df["Ri_Global_Risk"]
    df["Context_Contribution"] = (1 - lambda_val) * df["Ci_Context_Score"]
    df["Lambda"] = lambda_val

    # st.markdown("<h1 class='main-header'>Dashboard</h1>", unsafe_allow_html=True)
    st.title("CADEMAS-ML")

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Models", "Context", "About"])

    # --- TAB 1 ---
    with tab1:
        c01, c0, c1, c2, c3, c4 = st.columns(6)
        c01.metric("Positive label", f"{st.session_state.p_label}")
        c0.metric(label="Cases (n)", value=f"{len(df['Final_Score'])}")
        c1.metric(label="Avg. Prior. Score", value=f"{df['Final_Score'].mean():.1%}")
        c2.metric("Avg. Global Risk (Ri)", f"{df['Ri_Global_Risk'].mean():.1%}")
        c3.metric("Avg. Context Align. (Ci)", f"{df['Ci_Context_Score'].mean():.1%}")
        c4.metric("High Priority Cases (> 0.8)", len(df[df["Final_Score"] > 0.8]))

        st.divider()

        g1, g2 = st.columns([1.5, 1])
        with g1:
            st.subheader("Prioritization: Risk vs Context")
            # Scatter Plot con Altair
            scatter = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('Ri_Global_Risk', title='ML Risk Score Ri (0–1)'),
                y=alt.Y('Ci_Context_Score', title='Context Alignment Ci (0–1)'),
                color=alt.Color('Final_Score', scale=alt.Scale(scheme='turbo'), title='Prior. Score'),
                tooltip=['Ri_Global_Risk', 'Ci_Context_Score', 'Final_Score']
            ).interactive()
            st.altair_chart(scatter, width='stretch')

        with g2:
            st.subheader("Prioritization Score")
            # Histograma Mejorado con Altair
            hist = alt.Chart(df).mark_bar().encode(
                x=alt.X("Final_Score", bin=alt.Bin(step=0.1), title="Global Score Range"),
                y=alt.Y('count()', title='Number of Cases'),
                color=alt.value("#1E88E5")
            )
            st.altair_chart(hist, width='stretch')

        st.subheader("Prioritized Cases")

        # 1. Definimos las columnas a mostrar
        id_col = "CaseID"
        prob_cols = [c for c in df.columns if c.endswith("_prob")]
        # Columnas calculadas (en orden lógico)
        calculated_cols = (
            [id_col] +
            ["Final_Score", "Ri_Global_Risk", "Ci_Context_Score",
             "Lambda", "ML_Contribution", "Context_Contribution"] +
            prob_cols
        )

        # Columnas originales (preservando su orden)
        original_cols = [c for c in st.session_state.master_data.columns if c not in calculated_cols]

        # Orden final: calculadas primero, luego dataset original completo
        final_cols = calculated_cols + original_cols

        # 2. Aplicamos Estilos con Pandas (Heatmap)
        cols_main = ["Final_Score", "Ri_Global_Risk", "Ci_Context_Score"]

        styled_df = df[final_cols].sort_values("Final_Score", ascending=False).style \
            .format("{:.1%}", subset=cols_main) \
            .background_gradient(cmap='RdYlGn_r', subset=['Final_Score'], vmin=0, vmax=1)
        # Nota: RdYlGn_r pone Verde en 0 (Bajo riesgo) y Rojo en 1 (Alto riesgo)

        # 3. Renderizamos con Column Config para toques extra
        st.dataframe(
            styled_df,
            column_config={
                "Final_Score": st.column_config.NumberColumn(
                    "Prioritization Score",
                    help="Weighted final prioritization score",
                    format="percent",
                ),
                "Ri_Global_Risk": st.column_config.ProgressColumn(
                    "ML Risk (Ri)",
                    format="percent",
                    min_value=0,
                    max_value=1,
                    color="auto-inverse"
                ),
                "Ci_Context_Score": st.column_config.ProgressColumn(
                    "Context Alignment (Ci)",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                    color="auto-inverse"
                ),
            },
            width='stretch',
            height=500
        )

    # --- TAB 2 ---
    with tab2:
        st.subheader("Model Weights")
        if st.session_state.ml_details:
            weights = st.session_state.ml_details["weights"]
            w_df = pd.DataFrame(list(weights.items()), columns=["Model", "Weight (Wi)"])
            st.dataframe(w_df, width='stretch')
        st.subheader("ADM Risk Probabilities")
        id_col = "CaseID"
        prob_cols = [c for c in df.columns if c.endswith("_prob")]

        # Construimos tabla extendida de probabilidades + riesgo global
        weights = st.session_state.ml_details["weights"]

        prob_df = df[[id_col] + prob_cols].copy()

        # Calcular Riesgo Global explícito (suma ponderada)
        global_risk = np.zeros(len(prob_df))
        for col in prob_cols:
            model_name = col.replace("_prob", "")
            # Buscamos el peso correspondiente (por nombre de modelo)
            for w_key, w_val in weights.items():
                if w_key.startswith(model_name):
                    global_risk += prob_df[col].values * w_val
                    break

        prob_df["Ri_Global_Risk"] = global_risk

        styled_prob_df = prob_df.style \
            .format("{:.1%}", subset=["Ri_Global_Risk"]) \
            .background_gradient(cmap='RdYlGn_r', subset=["Ri_Global_Risk"], vmin=0, vmax=1)

        st.dataframe(
            styled_prob_df,
            column_config={
                "Ri_Global_Risk": st.column_config.ProgressColumn(
                    "Global ML Risk (Ri)",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                )
            },
            width='stretch'
        )

    # --- TAB 3 (Visualización Difusa con Altair) ---
    with tab3:
        st.subheader("Membership Functions")

        # 1. Selector de Regla Inteligente
        rules = st.session_state.context_config['rules']

        # Diccionario para nombres bonitos en la UI
        type_labels = {
            'triangular': 'Triángulo',
            'trapezoidal': 'Trapecio',
            'linear_increasing': 'Lineal creciente',
            'linear_decreasing': 'Lineal decreciente'
        }

        # Generamos la lista de opciones formateadas
        rule_options = []
        for i, r in enumerate(rules):
            r_type = r.get('type')
            r_feat = r.get('feature')
            r_name = r.get('name')

            if r_type in ('categorical_map', 'categorical_set'):
                clean_type = 'Categorical'
            else:
                clean_type = type_labels.get(r_type, r_type)

            label = r_name if r_name else r_feat
            rule_options.append(f"{label} [{clean_type}]")

        selected_rule_idx = st.selectbox(
            "Select variable / rule to inspect:",
            range(len(rules)),
            format_func=lambda i: rule_options[i]
        )

        # Recuperamos la regla seleccionada
        selected_rule = rules[selected_rule_idx]
        feat = selected_rule.get('feature')
        m_type = selected_rule.get('type')
        params = selected_rule.get('params')
        rule_alias = selected_rule.get('name') or feat

        # 2. Verificar datos y preparar visualización
        raw_data = st.session_state.master_data

        if feat in raw_data.columns and m_type not in ('categorical_map', 'categorical_set'):

            # --- Lógica de Renderizado Altair ---

            # A. Preparar Rango X
            data_vals = raw_data[feat].dropna()
            min_d, max_d = data_vals.min(), data_vals.max()

            # El rango del gráfico debe cubrir los datos Y los parámetros de la regla
            # Concatenamos params con min/max de los datos para encontrar los límites
            all_points = list(params) + [min_d, max_d]
            x_start = min(all_points) * 0.95
            x_end = max(all_points) * 1.05

            x_grid = np.linspace(x_start, x_end, 300)

            # B. Calcular Y (Membresía) usando la función genérica
            y_grid = get_membership(x_grid, m_type, params)

            line_df = pd.DataFrame({'x_val': x_grid, 'membership': y_grid})

            # C. Preparar Líneas Verticales de Referencia
            labels = []
            if m_type == 'triangular':
                labels = ['a (Inicio)', 'b (Pico)', 'c (Fin)']
            elif m_type == 'trapezoidal':
                labels = ['a (Inicio)', 'b (Plano in)', 'c (Plano out)', 'd (Fin)']
            elif m_type == 'linear_increasing':
                labels = ['a (Base 0)', 'b (Tope 1)']
            elif m_type == 'linear_decreasing':
                labels = ['a (Tope 1)', 'b (Base 0)']
            else:
                labels = [f"p{i}" for i in range(len(params))]

            rules_df = pd.DataFrame({
                'x_pos': params,
                'label': labels,
                'color': ['red'] * len(params)
            })

            # --- D. Construcción del Gráfico (Altair) ---

            # Capa 1: Histograma
            hist = alt.Chart(raw_data).mark_bar(color='#e0e0e0', opacity=0.7).encode(
                x=alt.X(feat, bin=alt.Bin(maxbins=30), title=feat),
                y=alt.Y('count()', title='Frequency'),
                tooltip=['count()']
            )

            # Capa 2: Curva Membresía
            line = alt.Chart(line_df).mark_line(color="#FF4B4B", strokeWidth=3).encode(
                x='x_val',
                y=alt.Y('membership', title='Membership (μ)', scale=alt.Scale(domain=[0.0, 1])),
                tooltip=[alt.Tooltip('x_val', format='.2f'), alt.Tooltip('membership', format='.2f')]
            )

            # Capa 3: Referencias Verticales
            refs = alt.Chart(rules_df).mark_rule(strokeDash=[5, 5],  strokeWidth=2, color='black', opacity=0.5).encode(
                x='x_pos',
                tooltip=['label', 'x_pos']
            )

            # Combinar con ejes independientes
            final_chart = alt.layer(hist, line, refs).resolve_scale(
                y='independent'
            ).properties(
                height=350,
                title=f"Membership and Frequency for '{rule_alias}'"
            )

            st.altair_chart(final_chart, width='stretch')

        elif feat in raw_data.columns and m_type in ('categorical_map', 'categorical_set'):
            st.info(f"Categorical rule '{rule_alias}' selected. Membership curves are only available for numeric rules.")
        else:
            st.warning(f"Column '{feat}' is not present in the dataset.")

        st.subheader("Numerical Audit")

        # Tabla detallada
        feature_name = f"mu_{rule_alias}"
        audit_cols = [feat, feature_name]
        if feature_name in st.session_state.fuzzy_details.columns:
            id_col = "CaseID"

            audit_df = pd.concat([
                raw_data[[id_col, feat]].reset_index(drop=True),
                st.session_state.fuzzy_details[[feature_name]].reset_index(drop=True)
            ], axis=1)

            styled_context_df = audit_df.style \
                .format("{:.1%}", subset=[feature_name]) \
                .background_gradient(cmap='RdYlGn_r', subset=[feature_name], vmin=0, vmax=1)

            st.dataframe(
                styled_context_df,
                column_config={
                    feature_name: st.column_config.ProgressColumn(
                        f"μ({rule_alias})",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    )
                },
                width='stretch'
            )

        # --- Derived Rules Visualization ---
        st.subheader("Derived Rules Overview")
        derived_rules = st.session_state.context_config.get('derived_rules', [])
        if derived_rules:
            derived_cols = [r['name'] for r in derived_rules]
            derived_mu_cols = [f"mu_{c}" for c in derived_cols if f"mu_{c}" in st.session_state.fuzzy_details.columns]
            if derived_mu_cols:
                heatmap_df = st.session_state.fuzzy_details[derived_mu_cols].copy()
                st.dataframe(
                    heatmap_df.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=1),
                    width='stretch',
                    height=400
                )

                # Optional: scatterplot of two selected derived rules
                if len(derived_mu_cols) >= 2:
                    st.subheader("Scatterplot of Derived Rules")
                    col1, col2 = st.columns(2)
                    with col1:
                        x_rule = st.selectbox("X-axis derived rule", derived_mu_cols, index=0)
                    with col2:
                        y_rule = st.selectbox("Y-axis derived rule", derived_mu_cols, index=1)

                    scatter_df = st.session_state.fuzzy_details[[x_rule, y_rule]].copy()
                    scatter_df['CaseID'] = st.session_state.master_data['CaseID']
                    scatter_plot = alt.Chart(scatter_df).mark_circle(size=60).encode(
                        x=alt.X(x_rule, title=x_rule),
                        y=alt.Y(y_rule, title=y_rule),
                        color=alt.Color('CaseID', scale=alt.Scale(scheme='turbo')),
                        tooltip=['CaseID', x_rule, y_rule]
                    ).interactive()
                    st.altair_chart(scatter_plot, width='stretch', use_container_width=True)
            else:
                st.info("Derived rules exist but no computed values are available.")
        else:
            st.info("No derived rules defined in the context configuration.")

# --- TAB 4 ---
    with tab4:
        st.subheader("About")

        st.markdown(
            """
            **Authors:**
            - Pavel Novoa Hernández (Universidad de La Laguna) — pnovoahe@ull.edu.es
            - David A. Pelta (Universidad de Granada) — dpelta@ugr.es
            - Mariia Godz (Universidad de Granada) — mariiagodz@ugr.es

            This application has been funded by the project:

            *Study, Analysis and Evaluation of Cooperative Automated Decision-Making Systems (CADEMAS)* \
            (reference number **PID2023-146575NB-I00**), funded by **MCIU/AEI/10.13039/501100011033** and by **FSE+**.
            """
        )

else:
    st.title("CADEMAS-ML")
    st.subheader("Welcome to CADEMAS-ML, a cooperative and context-aware decision support system.")
    st.info("👈 Upload the required files to start the analysis.")
    render_animated_header()

# --- INSTRUCCIONES PARA PARCHE ---
# Este parche elimina la selección de columna de ID por parte del usuario,
# y en su lugar genera una columna "CaseID" de forma automática y consecutiva (1, 2, ...).
# Cambia todas las referencias a la columna de ID para que usen "CaseID".
# El usuario solo debe subir el archivo CSV y el sistema se encargará del identificador único.
