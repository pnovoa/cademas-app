import streamlit as st
import h2o
import pandas as pd
import numpy as np
import json
import os
import tempfile
import altair as alt
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cooperative, Context-aware DSS", layout="wide", page_icon="🧠")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .main-header {font-size: 2rem; color: #1E88E5;}
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #1E88E5;}
</style>
""", unsafe_allow_html=True)


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
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=50)
    st.title("Configuración")

    with st.expander("📂 1. Archivos y Datos", expanded=True):
        json_ml = st.file_uploader("Config. Modelos (JSON)", type=['json'])
        json_context = st.file_uploader("Config. Contexto (JSON)", type=['json'])
        model_files = st.file_uploader("Modelos MOJO (.zip)", type=['zip'], accept_multiple_files=True)
        data_file = st.file_uploader("Dataset (.csv)", type=['csv'])

    with st.expander("⚙️ 2. Parámetros ML", expanded=False):
        selected_metric = None
        if json_ml:
            feature_config = json.load(json_ml)
            first_key = list(feature_config.keys())[0]
            metrics = list(feature_config[first_key].get("performance", {}).keys())
            selected_metric = st.selectbox("Métrica de Peso (Wi)", metrics)

    with st.expander("🧠 3. Parámetros Difusos", expanded=False):
        aggregation_method = st.selectbox("Agregación Contexto", ["average", "minimum (strict)", "product"])

    run_calc = st.button("🚀 EJECUTAR ANÁLISIS", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### Ajuste de Decisión")
    lambda_val = st.slider("⚖️ Lambda (Peso ML)", 0.0, 1.0, 0.5, 0.05)
    st.caption(f"ML: {lambda_val:.0%} | Contexto: {1 - lambda_val:.0%}")

# --- 4. ESTADO ---
if 'base_results' not in st.session_state: st.session_state.base_results = None
if 'fuzzy_details' not in st.session_state: st.session_state.fuzzy_details = None
if 'ml_details' not in st.session_state: st.session_state.ml_details = None
if 'context_config' not in st.session_state: st.session_state.context_config = None  # Guardamos config para graficar
if 'master_data' not in st.session_state: st.session_state.master_data = None  # Guardamos raw data para histogramas

# --- 5. EJECUCIÓN ---
if run_calc:
    if feature_config and model_files and data_file and selected_metric and json_context:
        with st.spinner("🧠 Procesando..."):
            try:
                # A. Carga
                master_df = pd.read_csv(data_file)
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
                    cols = feature_config[m_file.name]["features"]
                    hf = h2o.H2OFrame(master_df[cols])
                    try:
                        mojo = h2o.import_mojo(path)
                        preds = mojo.predict(hf).as_data_frame()
                        p_col = 'p1' if 'p1' in preds.columns else preds.columns[-1]
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

                st.success("Cálculo finalizado.")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("Faltan archivos por subir.")

# --- 6. VISUALIZACIÓN ---

if st.session_state.base_results is not None:
    df = st.session_state.base_results.copy()
    df["Final_Score"] = (lambda_val * df["Ri_Global_Risk"]) + ((1 - lambda_val) * df["Ci_Context_Score"])

    st.markdown("<h1 class='main-header'>Dashboard de Decisión</h1>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Resumen Ejecutivo", "🤖 Auditoría ML", "🧠 Auditoría Contexto"])

    # --- TAB 1 ---
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Riesgo Global (Promedio)", f"{df['Ri_Global_Risk'].mean():.1%}")
        c2.metric("Ajuste Contexto (Promedio)", f"{df['Ci_Context_Score'].mean():.1%}")
        c3.metric("Score Final (Promedio)", f"{df['Final_Score'].mean():.1%}")
        c4.metric("Alto Riesgo (n, > 0.8)", len(df[df["Final_Score"] > 0.8]))

        st.divider()

        g1, g2 = st.columns([2, 1])
        with g1:
            st.subheader("Mapa de Riesgo vs Contexto")
            # Scatter Plot con Altair
            scatter = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('Ri_Global_Risk', title='Riesgo ML (0-1)'),
                y=alt.Y('Ci_Context_Score', title='Cumplimiento Contexto (0-1)'),
                color=alt.Color('Final_Score', scale=alt.Scale(scheme='turbo'), title='Score Final'),
                tooltip=['Ri_Global_Risk', 'Ci_Context_Score', 'Final_Score']
            ).interactive()
            st.altair_chart(scatter, use_container_width=True)

        with g2:
            st.subheader("Distribución Score Final")
            # Histograma Mejorado con Altair
            hist = alt.Chart(df).mark_bar().encode(
                x=alt.X("Final_Score", bin=alt.Bin(step=0.1), title="Rango de Score Final"),
                y=alt.Y('count()', title='Frecuencia de Casos'),
                color=alt.value("#1E88E5")
            )
            st.altair_chart(hist, use_container_width=True)

        st.subheader("Tabla de Resultados")
        cols_show = ["Final_Score", "Ri_Global_Risk", "Ci_Context_Score"] + [c for c in df.columns if
                                                                             "prob" not in c and "Score" not in c and "Risk" not in c]
        st.dataframe(df.sort_values("Final_Score", ascending=False).head(50), column_order=cols_show,
                     use_container_width=True)

    # --- TAB 2 ---
    with tab2:
        st.subheader("Pesos del Ensemble")
        if st.session_state.ml_details:
            weights = st.session_state.ml_details["weights"]
            w_df = pd.DataFrame(list(weights.items()), columns=["Modelo", "Peso (Wi)"])
            st.dataframe(w_df, use_container_width=True)
        st.subheader("Probabilidades Crudas")
        st.dataframe(df[[c for c in df.columns if c.endswith("_prob")]], use_container_width=True)

    # --- TAB 3 (Visualización Difusa con Altair) ---
    with tab3:
        st.subheader("Visualización Interactiva de Funciones de Membresía")

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
            "Seleccionar Variable/Regla para inspeccionar:",
            range(len(rules)),
            format_func=lambda i: rule_options[i]  # Muestra el string bonito
        )

        # Recuperamos la regla seleccionada
        selected_rule = rules[selected_rule_idx]
        feat = selected_rule['feature']
        m_type = selected_rule['type']  # <--- Tipo dinámico
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
                x=alt.X(feat, bin=alt.Bin(maxbins=40), title=feat),
                y=alt.Y('count()', title='Frecuencia'),
                tooltip=['count()']
            )

            # Capa 2: Curva Membresía
            line = alt.Chart(line_df).mark_line(color='#1E88E5', strokeWidth=3).encode(
                x='x_val',
                y=alt.Y('membership', title='Membresía (μ)', scale=alt.Scale(domain=[0.0, 1])),
                tooltip=[alt.Tooltip('x_val', format='.2f'), alt.Tooltip('membership', format='.2f')]
            )

            # Capa 3: Referencias Verticales
            refs = alt.Chart(rules_df).mark_rule(strokeDash=[5, 5], color='red', opacity=0.5).encode(
                x='x_pos',
                tooltip=['label', 'x_pos']
            )

            # Etiquetas de texto para las referencias (Opcional, pero útil)
            text_refs = alt.Chart(rules_df).mark_text(align='left', dy=-10, color='red').encode(
                x='x_pos',
                text='label'
            )

            # Combinar con ejes independientes
            final_chart = alt.layer(hist, line, refs, text_refs).resolve_scale(
                y='independent'
            ).properties(
                height=350,
                title=f"Regla {clean_type} para '{feat}'"
            )

            st.altair_chart(final_chart, use_container_width=True)

        else:
            st.warning(f"La columna {feat} no está en el dataset.")

        st.divider()
        st.subheader("Auditoría Numérica")

        # Tabla detallada
        audit_cols = [feat, f"mu_{feat}"]
        if f"mu_{feat}" in st.session_state.fuzzy_details.columns:
            audit_df = pd.concat([
                raw_data[[feat]].reset_index(drop=True),
                st.session_state.fuzzy_details[[f"mu_{feat}"]].reset_index(drop=True)
            ], axis=1)
            st.dataframe(audit_df.head(100), use_container_width=True)

else:
    st.info("👈 Sube los archivos para comenzar.")