from traceback import print_last

import streamlit as st
import h2o
import pandas as pd
import numpy as np
import json
import os
import tempfile
import altair as alt
from animation import render_animated_header
from h2o.model import ModelBase
from fuzzy_context import calculate_context_score, get_membership

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
            if "selected_metric" not in st.session_state:
                st.session_state.selected_metric = metrics[0] if metrics else None
            st.selectbox(
                "Metric used for model weighting (w)",
                metrics,
                key="selected_metric"
            )
            selected_metric = st.session_state.selected_metric

    with st.expander("3. Context Parameters", expanded=False):
        if "aggregation_method" not in st.session_state:
            st.session_state.aggregation_method = "average"
        st.selectbox(
            "Context aggregation operator",
            ["average", "minimum (strict)", "product"],
            key="aggregation_method"
        )
        aggregation_method = st.session_state.aggregation_method

    run_calc = st.button("Run analysis", type="primary", width='stretch')

    st.markdown("## Decision Adjustment")
    lambda_val = st.slider("Lambda (weight)", 0.0, 1.0, 0.5, 0.01)
    st.caption(f"Context contribution: {1 - lambda_val:.0%} | Risk contribution: {lambda_val:.0%}")


# --- 4. ESTADO ---
if 'base_results' not in st.session_state: st.session_state.base_results = None
if 'fuzzy_details' not in st.session_state: st.session_state.fuzzy_details = None
if 'ml_details' not in st.session_state: st.session_state.ml_details = None
if 'context_config' not in st.session_state: st.session_state.context_config = None  # Guardamos config para graficar
if 'master_data' not in st.session_state: st.session_state.master_data = None  # Guardamos raw data para histogramas
if 'p_label' not in st.session_state: st.session_state.p_label = None


# --- 5. EJECUCIÓN ---
if run_calc:
    st.session_state["run_triggered"] = True
if st.session_state.get("run_triggered", False):
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

                st.session_state["run_triggered"] = False
                st.success("Computation completed successfully.")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("Required input files are missing.")

# --- 6. VISUALIZACIÓN ---

if st.session_state.base_results is not None:
    df = st.session_state.base_results.copy()
    df["Prioritization_Score "] = (lambda_val * df["Ri_Global_Risk"]) + ((1 - lambda_val) * df["Ci_Context_Score"])

    df["ML_Contribution"] = lambda_val * df["Ri_Global_Risk"]
    df["Context_Contribution"] = (1 - lambda_val) * df["Ci_Context_Score"]
    df["Lambda"] = lambda_val

    # st.markdown("<h1 class='main-header'>Dashboard</h1>", unsafe_allow_html=True)
    st.title("CADEMAS-ML")

    tab1, tab2, tab3, tab_decision, tab4 = st.tabs(["Overview", "Models", "Context", "Decision", "About"])

    # --- TAB 1 ---
    with tab1:
        c01, c0, c1, c2, c3, c4 = st.columns(6)
        c01.metric("Positive label", f"{st.session_state.p_label}")
        c0.metric(label="Cases (n)", value=f"{len(df['Prioritization_Score '])}")
        c1.metric(label="Avg. Prior. Score", value=f"{df['Prioritization_Score '].mean():.1%}")
        c2.metric("Avg. Global Risk (Ri)", f"{df['Ri_Global_Risk'].mean():.1%}")
        c3.metric("Avg. Context Align. (Ci)", f"{df['Ci_Context_Score'].mean():.1%}")
        c4.metric("High Priority Cases (> 0.75)", len(df[df["Prioritization_Score "] > 0.75]))

        st.divider()

        g1, g2 = st.columns([1.5, 1])
        with g1:
            st.subheader("Prioritization: Risk vs Context")
            # Scatter Plot con Altair
            scatter = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('Ri_Global_Risk', title='ML Risk Score Ri (0–1)'),
                y=alt.Y('Ci_Context_Score', title='Context Alignment Ci (0–1)'),
                color=alt.Color('Prioritization_Score ', scale=alt.Scale(scheme='turbo'), title='Prior. Score'),
                tooltip=['CaseID', 'Ri_Global_Risk', 'Ci_Context_Score', 'Prioritization_Score ']
            ).interactive()
            st.altair_chart(scatter, width='stretch')

        with g2:
            st.subheader("Prioritization Score")
            # Histograma Mejorado con Altair
            hist = alt.Chart(df).mark_bar().encode(
                x=alt.X("Prioritization_Score ", bin=alt.Bin(step=0.1), title="Global Score Range"),
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
            ["Prioritization_Score ", "Ri_Global_Risk", "Ci_Context_Score",
             "Lambda", "ML_Contribution", "Context_Contribution"] +
            prob_cols
        )

        # Columnas originales (preservando su orden)
        original_cols = [c for c in st.session_state.master_data.columns if c not in calculated_cols]

        # Orden final: calculadas primero, luego dataset original completo
        final_cols = calculated_cols + original_cols

        # 2. Aplicamos Estilos con Pandas (Heatmap)
        cols_main = ["Prioritization_Score ", "Ri_Global_Risk", "Ci_Context_Score"]

        styled_df = df[final_cols].sort_values("Prioritization_Score ", ascending=False).style \
            .format("{:.1%}", subset=cols_main) \
            .background_gradient(cmap='RdYlGn_r', subset=['Prioritization_Score '], vmin=0, vmax=1)
        # Nota: RdYlGn_r pone Verde en 0 (Bajo riesgo) y Rojo en 1 (Alto riesgo)

        # 3. Renderizamos con Column Config para toques extra
        st.dataframe(
            styled_df,
            column_config={
                "Prioritization_Score ": st.column_config.NumberColumn(
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
            'triangular': 'Triangle',
            'trapezoidal': 'Trapezoidal',
            'linear_increasing': 'Linear increasing',
            'linear_decreasing': 'Linear decreasing'
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

            if feat in raw_data.columns and m_type in ('categorical_map', 'categorical_set'):
                styled_context_df = audit_df.style
                st.dataframe(
                    styled_context_df,
                    width='stretch'
                )
            else:
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
                heatmap_df['Context_Alignment'] = st.session_state.base_results['Ci_Context_Score']
                heatmap_df.insert(
                    0,
                    "CaseID",
                    st.session_state.master_data["CaseID"].values
                )
                st.dataframe(
                    heatmap_df.style.background_gradient(cmap='RdYlGn_r', vmin=0, vmax=1,
                                                         subset=derived_mu_cols),
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

                    if x_rule != y_rule:
                        scatter_df = st.session_state.fuzzy_details[[x_rule, y_rule]].copy()
                        scatter_df['CaseID'] = st.session_state.master_data['CaseID']
                        scatter_df['Context_Alignment'] = st.session_state.base_results['Ci_Context_Score']
                        scatter_plot = alt.Chart(scatter_df).mark_circle(size=60).encode(
                            x=alt.X(x_rule, title=x_rule),
                            y=alt.Y(y_rule, title=y_rule),
                            color=alt.Color('Context_Alignment', scale=alt.Scale(scheme='turbo')),
                            tooltip=['CaseID', x_rule, y_rule]
                        ).interactive()
                        st.altair_chart(scatter_plot, width='stretch')
                    else:
                        st.warning("Please select different derived rules.")
            else:
                st.info("Derived rules exist but no computed values are available.")
        else:
            st.info("No derived rules defined in the context configuration.")


    # --- NUEVO TAB: DECISION (MEJORADO) ---
    with tab_decision:
        st.subheader("Sensitivity Analysis (Bump Chart)")
        st.markdown(f"""
        Visualize how the **Priority Ranking** changes when moving the weight ($\\lambda$) 
        from pure *Context Alignment* ($\\lambda = 0$) to pure *ML Risk* ($\\lambda = 1$).
        """)

        # A. CONTROLS
        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns(3)
        with c_ctrl1:
            n_partitions = st.slider("Lambda Partitions (Steps)", min_value=2, max_value=10, value=4)
        with c_ctrl2:
            top_n_show = st.slider("Show Top N Cases", min_value=5, max_value=50, value=15,
                                   help="Filters the cases with the highest ranking.")
        with c_ctrl3:
            select_by_current_lambda = st.checkbox(f"Select Cases with Current Lambda ($\\lambda = {lambda_val}$)", value=True,
                                                   help="""Whether the top N cases are selected based on the current lambda value (check) or 
                                                         based on the average ranks across all lambda partitions (uncheck).""")

        # B. DATA PREPARATION
        # Generate exact steps. E.g.: [0.0, 0.25, 0.5, 0.75, 1.0]
        lambda_steps = np.linspace(0, 1, n_partitions + 1)

        bump_data = []
        for l_step in lambda_steps:
            temp_df = df[[id_col, "Ri_Global_Risk", "Ci_Context_Score", "Prioritization_Score "]].copy()
            # Simulated Score
            temp_df["Sim_Score"] = (l_step * temp_df["Ri_Global_Risk"]) + (
                    (1 - l_step) * temp_df["Ci_Context_Score"])
            # Ranking (method='first' breaks ties by order of appearance)
            temp_df["Rank"] = temp_df["Sim_Score"].rank(ascending=False, method='first')
            temp_df["SelectRank"] = temp_df["Prioritization_Score "].rank(ascending=False, method='first') if select_by_current_lambda else temp_df["Rank"]
            temp_df["Lambda"] = l_step
            bump_data.append(temp_df)

        bump_df = pd.concat(bump_data)

        # C. FILTERING (TOP N)
        avg_ranks = bump_df.groupby(id_col)["SelectRank"].mean().sort_values()
        top_ids = avg_ranks.head(top_n_show).index.tolist()
        filtered_bump_df = bump_df[bump_df[id_col].isin(top_ids)]

        # D. ALTAIR CHART LAYERS

        # 1. Base Chart (Define common axes)
        # Note: On the X axis we force 'values' to show only the exact partitions.
        base = alt.Chart(filtered_bump_df).encode(
            x=alt.X('Lambda:Q',
                    axis=alt.Axis(values=list(lambda_steps), format='.2f', title="Lambda Weight (λ)"),
                    scale=alt.Scale(domain=[0, 1])
                    ),
            y=alt.Y('Rank:Q',
                    title='Ranking (1 = Highest Priority)',
                    scale=alt.Scale(reverse=True, zero=False),  # reverse=True puts 1 at the top
                    axis=alt.Axis(tickMinStep=1)  # Only integers on Y axis
                    ),
            color=alt.Color(f'{id_col}:N', legend=None)  # Remove side legend to use direct labels
        )

        # 2. Line Layer (Smooth interpolation)
        lines = base.mark_line(interpolate='monotone', strokeWidth=3).encode(
            tooltip=[
                alt.Tooltip(id_col, title="ID"),
                alt.Tooltip("Lambda", format=".2f"),
                alt.Tooltip("Rank", title="Ranking"),
                alt.Tooltip("Sim_Score", title="Score", format=".1%")
            ]
        )

        # 3. Points Layer (Big circles)
        # Using size=100 (or more) to make them bigger than the line
        points = base.mark_circle(size=120, opacity=1).encode(
            tooltip=[alt.Tooltip(id_col), alt.Tooltip("Rank")]
        )

        # 4. Left Labels (Lambda = 0)
        text_start = base.mark_text(align='right', dx=-12, fontSize=11).encode(
            text=f'{id_col}:N'
        ).transform_filter(
            (alt.datum.Lambda == 0.0)
        )

        # 5. Right Labels (Lambda = 1)
        text_end = base.mark_text(align='left', dx=12, fontSize=11).encode(
            text=f'{id_col}:N'
        ).transform_filter(
            (alt.datum.Lambda == 1.0)
        )

        # Combine everything
        final_chart = (lines + points + text_start + text_end).interactive()

        st.altair_chart(final_chart, width='stretch', theme="streamlit", height=500)

        st.info(f"Showing the top {len(top_ids)} ranked cases with above-average ranking.")

# --- TAB 4 ---
    with tab4:
        st.subheader("About")

        st.markdown(
            """
            **Authors:**
            - **Pavel Novoa Hernández** (Universidad de La Laguna, Spain) — pnovoahe@ull.edu.es
            - David A. Pelta (Universidad de Granada, Spain) — dpelta@ugr.es
            - Mariia Godz (Universidad de Granada, Spain) — mariiagodz@ugr.es

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

