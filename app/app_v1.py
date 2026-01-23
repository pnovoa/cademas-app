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
    """Router que selecciona la función matemática correcta."""
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


def calculate_context_score(df, context_config, aggregation):
    fuzzy_scores = pd.DataFrame()
    for rule in context_config['rules']:
        col = rule['feature']
        m_type = rule['type']  # <--- Nuevo
        params = rule['params']  # <--- Lista dinámica

        if col in df.columns:
            # Llamamos al Router
            fuzzy_scores[f"mu_{col}"] = get_membership(df[col].values, m_type, params)
        else:
            fuzzy_scores[f"mu_{col}"] = 0.0

    # ... (el resto de la agregación sigue igual: average, min, prod) ...
    if aggregation == "average":
        scores = fuzzy_scores.mean(axis=1)
    elif aggregation == "minimum (strict)":
        scores = fuzzy_scores.min(axis=1)
    elif aggregation == "product":
        scores = fuzzy_scores.prod(axis=1)
    else:
        scores = fuzzy_scores.mean(axis=1)

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

    tab1, tab2, tab3 = st.tabs(["Overview", "Models", "Context"])

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
            clean_type = type_labels.get(r['type'], r['type'])
            rule_options.append(f"{r['feature']} [{clean_type}]")

        selected_rule_idx = st.selectbox(
            "Select variable / rule to inspect:",
            range(len(rules)),
            format_func=lambda i: rule_options[i]
        )

        # Recuperamos la regla seleccionada
        selected_rule = rules[selected_rule_idx]
        feat = selected_rule['feature']
        m_type = selected_rule['type']  # <--- Tipo dinámico
        s_type = type_labels.get(m_type)
        params = selected_rule['params']  # <--- Lista dinámica (longitud 2, 3 o 4)

        # 2. Verificar datos y preparar visualización
        raw_data = st.session_state.master_data

        if feat in raw_data.columns:

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
            # (Asegúrate de tener la función 'get_membership' definida como vimos antes)
            y_grid = get_membership(x_grid, m_type, params)

            line_df = pd.DataFrame({'x_val': x_grid, 'membership': y_grid})

            # C. Preparar Líneas Verticales de Referencia
            # Definimos etiquetas según la cantidad de parámetros
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
                title=f"Membership and Frequency for '{feat}'"
            )

            st.altair_chart(final_chart, width='stretch')

        else:
            st.warning(f"Column '{feat}' is not present in the dataset.")

        st.subheader("Numerical Audit")

        # Tabla detallada
        feature_name = f"mu_{feat}"
        audit_cols = [feat, feature_name]
        if f"mu_{feat}" in st.session_state.fuzzy_details.columns:
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
                        f"μ({feat})",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    )
                },
                width='stretch'
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
