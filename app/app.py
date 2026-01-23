import streamlit as st
import h2o
import pandas as pd
import numpy as np
import json
import os
import tempfile
import altair as alt
import streamlit.components.v1 as components

st.set_page_config(page_title="CADEMAS-ML", layout="wide")

# --- CSS PERSONALIZADO ---
custom_css = """
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:11pt;
    align-items:center;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# --- HEADER ANIMADO ---
def render_animated_header():
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@100..900&display=swap');
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
        .node-rect { fill: #1f2937; stroke: #374151; stroke-width: 2px; rx: 6px; }
        .text-title { fill: #f3f4f6; font-size: 12px; font-weight: semibold; font-family: 'Geist Mono', monospace; pointer-events: none; }
        .text-sub { fill: #9ca3af; font-size: 10px; font-family: 'Geist Mono', monospace; pointer-events: none; }
        .stroke-ml { stroke: #3b82f6; }
        .stroke-fuzzy { stroke: #f97316; }
        .stroke-hybrid { stroke: #8b5cf6; }
        .path-line { fill: none; stroke: #4b5563; stroke-width: 2px; opacity: 0.3; }
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
                    <text x="40" y="20" text-anchor="middle" dominant-baseline="middle" class="text-title">DATOS</text>
                    <text x="40" y="32" text-anchor="middle" class="text-sub">CSV</text>
                </g>
                <g transform="translate(10, 90)">
                    <rect width="80" height="40" class="node-rect" />
                    <text x="40" y="20" text-anchor="middle" dominant-baseline="middle" class="text-title">REGLAS</text>
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
                    <animateMotion dur="2s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear"><mpath href="#p1"/></animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2s" repeatCount="indefinite" />
                </circle>
                <circle r="4" class="dot dot-fuzzy">
                    <animateMotion dur="2.5s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear"><mpath href="#p2"/></animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2.5s" repeatCount="indefinite" />
                </circle>
                <circle r="4" class="dot dot-ml">
                    <animateMotion dur="2s" begin="1s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear"><mpath href="#p3"/></animateMotion>
                     <animate attributeName="opacity" values="0;1;1;0" dur="2s" begin="1s" repeatCount="indefinite" />
                </circle>
                <circle r="4" class="dot dot-fuzzy">
                    <animateMotion dur="2.5s" begin="1.2s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear"><mpath href="#p4"/></animateMotion>
                     <animate attributeName="opacity" values="0;1;1;0" dur="2.5s" begin="1.2s" repeatCount="indefinite" />
                </circle>
                 <circle r="5" class="dot dot-hybrid">
                    <animateMotion dur="3s" begin="0.5s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear"><mpath href="#p5"/></animateMotion>
                     <animate attributeName="opacity" values="0;1;1;0" dur="3s" begin="0.5s" repeatCount="indefinite" />
                </circle>
            </svg>
        </div>
    </body>
    </html>
    """
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


@st.cache_data
def load_csv_preview(file):
    """Carga solo las columnas para previsualización rápida"""
    return pd.read_csv(file, nrows=0).columns.tolist()


@st.cache_data
def load_full_csv(file):
    return pd.read_csv(file)


# --- MOTOR DIFUSO ---
def get_membership(x, m_type, params):
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
    if a == b or b == c: return np.zeros_like(x)
    term1 = (x - a) / (b - a);
    term2 = (c - x) / (c - b)
    return np.maximum(0, np.minimum(term1, term2))


def trapezoidal(x, a, b, c, d):
    term1 = (x - a) / (b - a) if b > a else np.ones_like(x)
    term3 = (d - x) / (d - c) if d > c else np.ones_like(x)
    return np.maximum(0, np.minimum(np.minimum(term1, 1), term3))


def linear_increasing(x, a, b):
    if a == b: return (x >= a).astype(float)
    y = (x - a) / (b - a)
    return np.maximum(0, np.minimum(y, 1))


def linear_decreasing(x, a, b):
    if a == b: return (x <= a).astype(float)
    y = (b - x) / (b - a)
    return np.maximum(0, np.minimum(y, 1))


def calculate_context_score(df, context_config, aggregation):
    fuzzy_scores = pd.DataFrame()
    for rule in context_config['rules']:
        col = rule['feature']
        m_type = rule['type']
        params = rule['params']
        if col in df.columns:
            fuzzy_scores[f"mu_{col}"] = get_membership(df[col].values, m_type, params)
        else:
            fuzzy_scores[f"mu_{col}"] = 0.0

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
    st.title("Configuración")

    with st.expander("1. Archivos y Datos", expanded=True):
        json_ml = st.file_uploader("Configuración ML (JSON)", type=['json'])
        json_context = st.file_uploader("Configuración Contexto (JSON)", type=['json'])
        model_files = st.file_uploader("Modelos MOJO (.zip)", type=['zip'], accept_multiple_files=True)
        data_file = st.file_uploader("Dataset (.csv)", type=['csv'])

    # --- CONFIGURACIÓN DE VARIABLES (Dentro de ML o separada) ---
    # Lo ponemos en un expander dedicado que se activa cuando hay datos
    case_id_selection = "Auto-ID"
    target_col_selection = "Ninguno"

    if data_file:
        cols = load_csv_preview(data_file)

        with st.expander("2. Definición de Variables", expanded=True):
            st.info("Configura las columnas clave del dataset.")

            # Selector ID con Fallback
            case_id_selection = st.selectbox(
                "Columna Identificador (ID)",
                ["(Auto) Generar Consecutivo"] + cols,
                index=0,
                help="Selecciona la columna que identifica a cada fila. Si no existe, usa 'Generar Consecutivo'."
            )

            # Selector Target
            target_col_selection = st.selectbox(
                "Columna Target (Ground Truth)",
                ["Ninguno"] + cols,
                index=0
            )

    with st.expander("3. Parámetros de ML", expanded=False):
        selected_metric = None
        if json_ml:
            feature_config = json.load(json_ml)
            first_key = list(feature_config.keys())[0]
            metrics = list(feature_config[first_key].get("performance", {}).keys())
            selected_metric = st.selectbox("Métrica para el peso (w)", metrics)

    with st.expander("4. Parámetros del contexto", expanded=False):
        aggregation_method = st.selectbox("Agregación Contexto", ["average", "minimum (strict)", "product"])

    run_calc = st.button("Ejecutar Análisis", type="primary", use_container_width=True)

    st.markdown("## Ajuste de Decisión")
    lambda_val = st.slider("Lambda (Peso)", 0.0, 1.0, 0.5, 0.01)
    st.caption(f"Riesgo: {lambda_val:.0%} | Contexto: {1 - lambda_val:.0%}")

# --- 4. ESTADO ---
if 'base_results' not in st.session_state: st.session_state.base_results = None
if 'fuzzy_details' not in st.session_state: st.session_state.fuzzy_details = None
if 'ml_details' not in st.session_state: st.session_state.ml_details = None
if 'context_config' not in st.session_state: st.session_state.context_config = None
if 'master_data' not in st.session_state: st.session_state.master_data = None
if 'id_col_name' not in st.session_state: st.session_state.id_col_name = "Case_ID"

# --- 5. EJECUCIÓN ---
if run_calc:
    if feature_config and model_files and data_file and selected_metric and json_context:
        with st.spinner("🧠 Procesando datos..."):
            try:
                # A. Carga
                master_df = load_full_csv(data_file)

                # B. LÓGICA DE FALLBACK DE ID
                final_id_col_name = "Case_ID"  # Nombre interno por defecto

                if case_id_selection == "(Auto) Generar Consecutivo":
                    # Generamos ID
                    master_df["Case_ID"] = range(1, len(master_df) + 1)
                    final_id_col_name = "Case_ID"
                else:
                    # Usamos la columna seleccionada
                    final_id_col_name = case_id_selection
                    # IMPORTANTE: Convertir a String para evitar problemas de tipos mixtos en Altair
                    master_df[final_id_col_name] = master_df[final_id_col_name].astype(str)

                # Guardamos el nombre real de la columna ID en sesión para usarlo luego
                st.session_state.id_col_name = final_id_col_name
                st.session_state.master_data = master_df

                context_config = json.load(json_context)
                st.session_state.context_config = context_config

                # C. ML Loop
                valid_models = [m.name for m in model_files if m.name in feature_config]
                metrics_vals = {m: feature_config[m]["performance"].get(selected_metric, 0) for m in valid_models}
                total = sum(metrics_vals.values())
                weights = {m: (val / total if total > 0 else 1 / len(valid_models)) for m, val in metrics_vals.items()}

                st.session_state.ml_details = {"weights": weights, "metric": selected_metric}

                risk_accum = np.zeros(len(master_df))
                temp_results = master_df.copy()

                for i, m_file in enumerate(model_files):
                    if m_file.name not in weights: continue
                    path = save_temp_file(m_file)
                    cols = feature_config[m_file.name]["features"]
                    hf = h2o.H2OFrame(master_df[cols])
                    try:
                        mojo = h2o.import_mojo(path)
                        preds = mojo.predict(hf).as_data_frame()

                        # Detectar columnas
                        label_col = 'predict'
                        p_col = 'p1' if 'p1' in preds.columns else preds.columns[-1]

                        vals = preds[p_col].values
                        labels = preds[label_col].values

                        risk_accum += vals * weights[m_file.name]

                        clean_name = m_file.name.split('.')[0]
                        temp_results[f"{clean_name}_label"] = labels
                        temp_results[f"{clean_name}_prob"] = vals
                    finally:
                        if os.path.exists(path): os.remove(path)
                        h2o.remove(hf)

                temp_results["Ri_Global_Risk"] = risk_accum

                # D. Contexto
                ci_scores, fuzzy_df = calculate_context_score(master_df, context_config, aggregation_method)
                temp_results["Ci_Context_Score"] = ci_scores
                temp_results = pd.concat([temp_results, fuzzy_df], axis=1)

                st.session_state.base_results = temp_results
                st.session_state.fuzzy_details = fuzzy_df
                st.success("Cálculo finalizado.")

            except Exception as e:
                st.error(f"Error crítico: {e}")
    else:
        st.error("Faltan archivos por subir.")

# --- 6. VISUALIZACIÓN ---

if st.session_state.base_results is not None:
    df = st.session_state.base_results.copy()
    id_col = st.session_state.id_col_name  # Recuperamos el nombre de la columna ID

    df["Final_Score"] = (lambda_val * df["Ri_Global_Risk"]) + ((1 - lambda_val) * df["Ci_Context_Score"])

    st.title("CADEMAS-ML")
    tab1, tab2, tab3 = st.tabs(["Resumen", "Modelos", "Contexto"])

    # --- TAB 1 ---
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score Final", f"{df['Final_Score'].mean():.1%}")
        c2.metric("Riesgo Global", f"{df['Ri_Global_Risk'].mean():.1%}")
        c3.metric("Ajuste Contexto", f"{df['Ci_Context_Score'].mean():.1%}")
        c4.metric("Score > 0.8", len(df[df["Final_Score"] > 0.8]))
        st.divider()

        g1, g2 = st.columns([1.5, 1])
        with g1:
            st.subheader("Mapa de Riesgo")

            # --- CORRECCIÓN CRÍTICA DE ALTAIR ---
            # Para evitar "Unable to determine data type", definimos explícitamente el tooltip
            # usando alt.Tooltip(..., type='nominal') para el ID.

            tooltips_def = [
                alt.Tooltip(id_col, title="ID Caso", type="nominal"),  # <--- AQUÍ ESTÁ EL FIX
                alt.Tooltip('Ri_Global_Risk', format='.2f'),
                alt.Tooltip('Ci_Context_Score', format='.2f'),
                alt.Tooltip('Final_Score', format='.2f')
            ]

            scatter = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('Ri_Global_Risk', title='Riesgo ML'),
                y=alt.Y('Ci_Context_Score', title='Contexto'),
                color=alt.Color('Final_Score', scale=alt.Scale(scheme='turbo')),
                tooltip=tooltips_def
            ).interactive()
            st.altair_chart(scatter, use_container_width=True)

        with g2:
            st.subheader("Distribución")
            hist = alt.Chart(df).mark_bar().encode(
                x=alt.X("Final_Score", bin=alt.Bin(step=0.1)),
                y="count()",
                color=alt.value("#1E88E5")
            )
            st.altair_chart(hist, use_container_width=True)

        st.subheader("Tabla de Resultados")

        # Orden de columnas inteligente
        cols_show = [id_col]
        if target_col_selection != "Ninguno" and target_col_selection in df.columns:
            cols_show.append(target_col_selection)

        cols_show.extend(["Final_Score", "Ri_Global_Risk", "Ci_Context_Score"])

        # Añadir resto (modelos, mu_)
        cols_show.extend(
            [c for c in df.columns if c not in cols_show and ("_prob" in c or "_label" in c or "mu_" in c)])

        st.dataframe(
            df[cols_show].sort_values("Final_Score", ascending=False).head(100),
            column_config={
                "Final_Score": st.column_config.NumberColumn("Score", format="%.2%"),
                "Ri_Global_Risk": st.column_config.ProgressColumn("Riesgo", min_value=0, max_value=1),
                "Ci_Context_Score": st.column_config.ProgressColumn("Contexto", min_value=0, max_value=1),
            },
            use_container_width=True,
            hide_index=True
        )

    # --- TAB 2 ---
    with tab2:
        st.subheader("Detalle Modelos")
        # Mostramos ID + columnas de modelos
        mod_cols = [id_col] + [c for c in df.columns if "_prob" in c or "_label" in c]
        st.dataframe(df[mod_cols], use_container_width=True, hide_index=True)

    # --- TAB 3 ---
    with tab3:
        st.subheader("Análisis Difuso")
        rules = st.session_state.context_config['rules']
        # (Selector de reglas simplificado...)
        sel_idx = st.selectbox("Regla:", range(len(rules)), format_func=lambda i: rules[i]['feature'])

        # ... (Código de gráfico de regla igual al anterior) ...
        # (Omitido por brevedad, es igual al bloque anterior pero asegurando mostrar el ID en la tabla abajo)

        st.divider()
        feat = rules[sel_idx]['feature']
        st.subheader(f"Auditoría: {feat}")

        audit_cols = [id_col, feat, f"mu_{feat}"]
        if target_col_selection != "Ninguno" and target_col_selection in df.columns:
            audit_cols.append(target_col_selection)

        st.dataframe(df[audit_cols].head(100), use_container_width=True, hide_index=True)

else:
    st.title("CADEMAS-ML")
    st.info("👈 Configura los datos en el menú izquierdo.")
    render_animated_header()