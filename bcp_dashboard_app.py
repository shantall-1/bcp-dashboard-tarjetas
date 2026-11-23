
import argparse
import os
import numpy as np
import pandas as pd

SEED = 42
RNG = np.random.default_rng(SEED)


LINEA_BASE = {
    "Centro":     dict(ventas=620, activas=465, pendientes=95),
    "San Isidro": dict(ventas=560, activas=420, pendientes=100),
    "Miraflores": dict(ventas=510, activas=347, pendientes=105),
    "Norte":      dict(ventas=480, activas=336, pendientes=95),
    "Sur":        dict(ventas=450, activas=298, pendientes=97),
}
for a, d in LINEA_BASE.items():
    d["anuladas"] = d["ventas"] - d["activas"] - d["pendientes"]

# Metas PROPUESTAS (supuestos explícitos, documentados en el plan de acción)
METAS = {
    "activacion": 0.75,   # meta de tasa de activación por cohorte
    "pendientes": 0.15,   # techo de pendientes de recojo
    "anulacion":  0.08,   # techo de anulación
    "umbral_alerta": 0.70 # umbral provisional de alerta (<70% activación)
}
TOL = 0.05  # tolerancia ±5 p.p. para semáforo amarillo

# Supuestos económicos/operativos para optimización y Monte Carlo (académicos)
SUPUESTOS = {
    "costo_contacto": 3.50,       # S/ por contacto a cliente pendiente
    "presupuesto": 1500.00,       # S/ total del piloto de seguimiento
    "conv_base_contactado": 0.30, # probabilidad base de activar si se contacta
    "factor_mejora": {            # multiplicador por agencia (desempeño observado)
        "Centro": 1.20, "San Isidro": 1.15, "Miraflores": 0.95,
        "Norte": 1.00, "Sur": 0.90},
    "n_sim": 10000,
    "meta_incremental": 40,      # activaciones incrementales objetivo del piloto
}

CANALES = ["Ventanilla", "Digital", "Campaña"]
P_CANAL = [0.50, 0.30, 0.20]
TIPOS = ["Clásica", "Oro", "Platinum"]
SEGMENTOS = ["Mass", "Premium", "Empresas"]

FECHAS = pd.date_range("2026-09-01", "2026-09-07", freq="D").date


# ---------------------------------------------------------------------------
# 1. GENERACIÓN DE DATOS REPRODUCIBLE (nivel tarjeta + nivel colaborador-día)
# ---------------------------------------------------------------------------
def generar_datos():
    """Devuelve (df_tarjetas: 2.620 filas, df_registros: 175 filas).
    Coherentes con la línea base del documento."""
    # ---------- 1a. Nivel tarjeta (2.620 filas) ----------
    tarjetas = []
    for ag, d in LINEA_BASE.items():
        estados = (["Activa"] * d["activas"] + ["Pendiente"] * d["pendientes"]
                   + ["Anulada"] * d["anuladas"])
        estados = list(RNG.permutation(estados))
        for i, est in enumerate(estados):
            fid = RNG.choice(FECHAS)
            tarjetas.append(dict(
                id_tarjeta=f"{ag[:3].upper()}-{i+1:04d}",
                agencia=ag,
                colaborador=f"{ag[:3].upper()}-C0{RNG.integers(1, 6)}",
                fecha_venta=fid,
                estado=est,
                canal=RNG.choice(CANALES, p=P_CANAL),
                tipo_tarjeta=RNG.choice(TIPOS, p=[0.55, 0.35, 0.10]),
                segmento=RNG.choice(SEGMENTOS, p=[0.60, 0.30, 0.10]),
                linea_credito=int(RNG.choice([3000, 6000, 10000, 20000], p=[.4, .3, .2, .1])),
                antiguedad_dias=int(np.clip(RNG.normal(
                    {"Activa": 2.0, "Pendiente": 4.5, "Anulada": 3.0}[est], 1.2), 1, 7)),
            ))
    df_tarjetas = pd.DataFrame(tarjetas)

    # ---------- 1b. Nivel colaborador-día (175 = 5 agencias x 25 colabs... )
    # 25 colaboradores = 5 por agencia; 5 x 7 días x 5 agencias = 175 registros
    registros = []
    for ag, d in LINEA_BASE.items():
        # repartir las ventas de la agencia en 35 celdas (5 colabs x 7 días)
        base = d["ventas"] // 35
        resto = d["ventas"] - base * 35
        extra = np.zeros(35, dtype=int)
        extra[RNG.choice(35, size=resto, replace=False)] = 1
        k = 0
        for c in range(1, 6):
            for f in FECHAS:
                registros.append(dict(
                    agencia=ag, colaborador=f"{ag[:3].upper()}-C0{c}",
                    fecha=f, ventas=int(base + extra[k])))
                k += 1
    df_registros = pd.DataFrame(registros)
    return df_tarjetas, df_registros


# ---------------------------------------------------------------------------
# 2. SEMÁFOROS Y TABLA RESUMEN "CON TODO LO REQUERIDO"
# ---------------------------------------------------------------------------
def semaforo(valor, meta, tipo="maximo", tol=TOL):
    """tipo='maximo': verde si valor>=meta (p.ej. activación).
       tipo='minimo' : verde si valor<=meta (p.ej. pendientes, anulación)."""
    ok = (valor >= meta) if tipo == "maximo" else (valor <= meta)
    cerca = abs(valor - meta) <= tol
    if ok:
        return "🟢"
    if cerca:
        return "🟡"
    return "🔴"


def tabla_resumen(df):
    """Tabla consolidada por agencia: KPIs, metas, semáforos y alerta."""
    filas = []
    for ag, g in df.groupby("agencia"):
        v = len(g)
        act = int((g.estado == "Activa").sum())
        pen = int((g.estado == "Pendiente").sum())
        anu = int((g.estado == "Anulada").sum())
        ta, tp, tan = act / v, pen / v, anu / v
        alerta = "⚠️ REVISAR" if ta < METAS["umbral_alerta"] else "OK"
        filas.append(dict(
            Agencia=ag, Ventas=v, Activas=act,
            Activación=f"{ta:.1%}", Meta_act=f"{METAS['activacion']:.0%}",
            Sem_act=semaforo(ta, METAS["activacion"], "maximo"),
            Pendientes=pen, Pend_pct=f"{tp:.1%}", Meta_pend=f"{METAS['pendientes']:.0%}",
            Sem_pend=semaforo(tp, METAS["pendientes"], "minimo"),
            Anuladas=anu, Anul_pct=f"{tan:.1%}", Meta_anul=f"{METAS['anulacion']:.0%}",
            Sem_anul=semaforo(tan, METAS["anulacion"], "minimo"),
            Alerta=alerta))
    tot = dict(Agencia="TOTAL", Ventas=len(df),
               Activas=int((df.estado == "Activa").sum()),
               Activación=f"{(df.estado == 'Activa').mean():.1%}",
               Meta_act=f"{METAS['activacion']:.0%}",
               Sem_act=semaforo((df.estado == "Activa").mean(), METAS["activacion"], "maximo"),
               Pendientes=int((df.estado == "Pendiente").sum()),
               Pend_pct=f"{(df.estado == 'Pendiente').mean():.1%}",
               Meta_pend=f"{METAS['pendientes']:.0%}",
               Sem_pend=semaforo((df.estado == "Pendiente").mean(), METAS["pendientes"], "minimo"),
               Anuladas=int((df.estado == "Anulada").sum()),
               Anul_pct=f"{(df.estado == 'Anulada').mean():.1%}",
               Meta_anul=f"{METAS['anulacion']:.0%}",
               Sem_anul=semaforo((df.estado == "Anulada").mean(), METAS["anulacion"], "minimo"),
               Alerta="—")
    return pd.DataFrame(filas + [tot])


# ---------------------------------------------------------------------------
# 3. MODELO PREDICTIVO: probabilidad de activación (regresión logística)
#    Split temporal (train <= 04-sep, test > 04-sep) para evitar fuga de info
# ---------------------------------------------------------------------------
def modelo_predictivo(df):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, classification_report)

    data = df.copy()
    data["y"] = (data.estado == "Activa").astype(int)
    data["fecha_ord"] = pd.to_datetime(data["fecha_venta"]).map(
        pd.Timestamp.toordinal).astype(float)
    X = data[["antiguedad_dias", "linea_credito", "canal", "tipo_tarjeta",
              "segmento", "agencia", "fecha_ord"]]
    y = data["y"]
    fecha = pd.to_datetime(data["fecha_venta"])
    train, test = fecha <= "2026-09-04", fecha > "2026-09-04"

    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"),
                              ["canal", "tipo_tarjeta", "segmento", "agencia"])],
                            remainder="passthrough")
    model = Pipeline([("pre", pre),
                      ("clf", LogisticRegression(max_iter=1000, random_state=SEED))])
    model.fit(X[train], y[train])
    proba = model.predict_proba(X[test])[:, 1]
    pred = (proba >= 0.5).astype(int)

    metricas = dict(
        n_train=int(train.sum()), n_test=int(test.sum()),
        auc=roc_auc_score(y[test], proba),
        precision=precision_score(y[test], pred, zero_division=0),
        recall=recall_score(y[test], pred, zero_division=0),
        f1=f1_score(y[test], pred, zero_division=0),
        matriz_confusion=confusion_matrix(y[test], pred).tolist(),
        reporte=classification_report(y[test], pred, zero_division=0,
                                      target_names=["No activa", "Activa"]),
        probas_test=proba, y_test=y[test].values)
    return model, metricas


def top_variables(model):
    """Coeficientes más influyentes (explicabilidad básica del modelo)."""
    clf = model.named_steps["clf"]
    pre = model.named_steps["pre"]
    nombres = list(pre.get_feature_names_out())
    coef = clf.coef_[0]
    imp = pd.DataFrame({"variable": nombres, "coef": coef})
    imp["abs"] = imp.coef.abs()
    return imp.sort_values("abs", ascending=False).head(10).drop(columns="abs")


# ---------------------------------------------------------------------------
# 4. OPTIMIZACIÓN: asignación de contactos de seguimiento (prog. lineal)
#    Maximiza activaciones incrementales esperadas sujeto a presupuesto.
# ---------------------------------------------------------------------------
def optimizacion(df):
    from scipy.optimize import linprog
    pend = df[df.estado == "Pendiente"].groupby("agencia").size().reindex(
        LINEA_BASE.keys()).fillna(0).astype(int).values
    conv = np.array([SUPUESTOS["conv_base_contactado"] *
                     SUPUESTOS["factor_mejora"][a] for a in LINEA_BASE])
    # max sum(conv_i * x_i)  ->  min -conv_i * x_i
    res = linprog(c=-conv,
                  A_ub=[[SUPUESTOS["costo_contacto"]] * len(pend)],
                  b_ub=[SUPUESTOS["presupuesto"]],
                  bounds=[(0, p) for p in pend], method="highs")
    x = np.round(res.x).astype(int)
    out = pd.DataFrame({
        "agencia": list(LINEA_BASE.keys()),
        "pendientes": pend,
        "contactos_asignados": x,
        "prob_conversion": conv,
        "activaciones_esperadas": np.round(x * conv, 1),
        "costo_S": np.round(x * SUPUESTOS["costo_contacto"], 2)})
    out.loc["TOTAL"] = ["TOTAL", pend.sum(), x.sum(), np.nan,
                        round((x * conv).sum(), 1), round((x * SUPUESTOS["costo_contacto"]).sum(), 2)]
    return res, out


# ---------------------------------------------------------------------------
# 5. SIMULACIÓN MONTE CARLO (10.000 iteraciones) - decisiones bajo incertidumbre
# ---------------------------------------------------------------------------
def monte_carlo(n=SUPUESTOS["n_sim"]):
    rng = np.random.default_rng(SEED + 1)
    pend = {a: LINEA_BASE[a]["pendientes"] for a in LINEA_BASE}
    total_pend = sum(pend.values())
    inc = np.zeros(n)
    for a, p in pend.items():
        resp = rng.beta(8, 20, n)                      # tasa de respuesta al contacto
        conv = rng.beta(12, 28, n) * SUPUESTOS["factor_mejora"][a]  # conversión dado contacto
        inc += p * np.clip(resp, 0, 1) * np.clip(conv, 0, 1)
    p10, p50, p90 = np.percentile(inc, [10, 50, 90])
    prob_meta = float((inc >= SUPUESTOS["meta_incremental"]).mean())
    return pd.Series({"P10": round(p10, 1), "P50": round(p50, 1),
                      "P90": round(p90, 1),
                      "P(>=meta)": f"{prob_meta:.1%}",
                      "iteraciones": n, "pendientes_totales": total_pend})


# ---------------------------------------------------------------------------
# 6. DASHBOARD STREAMLIT (filtros, metas, semáforos, evidencias descargables)
# ---------------------------------------------------------------------------
def run_dashboard():
    import streamlit as st
    import plotly.express as px

    st.set_page_config(page_title="BCP | Dashboard Tarjetas", layout="wide")
    st.title("💳 BCP – De la venta a la activación")
    st.caption("Periodo 1–7 sep 2026 · Prototipo académico · Metas y umbrales = propuestas documentadas")

    @st.cache_data
    def cargar():
        return generar_datos()

    df, reg = cargar()
    df["fecha_venta"] = pd.to_datetime(df["fecha_venta"])

    # ---------------- Filtros (barra lateral) ----------------
    st.sidebar.header("🔎 Filtros")
    f_ag = st.sidebar.multiselect("Agencia", sorted(df.agencia.unique()),
                                  default=sorted(df.agencia.unique()))
    f_col = st.sidebar.multiselect("Colaborador", sorted(df.colaborador.unique()),
                                   default=sorted(df.colaborador.unique()))
    f_est = st.sidebar.multiselect("Estado", ["Activa", "Pendiente", "Anulada"],
                                   default=["Activa", "Pendiente", "Anulada"])
    f_fec = st.sidebar.date_input("Cohorte de venta (rango)", [FECHAS[0], FECHAS[-1]])
    d = df[df.agencia.isin(f_ag) & df.colaborador.isin(f_col)
           & df.estado.isin(f_est)
           & (df.fecha_venta >= pd.Timestamp(f_fec[0]))
           & (df.fecha_venta <= pd.Timestamp(f_fec[-1]))]

    # ---------------- KPIs con semáforo ----------------
    v = len(d); act = int((d.estado == "Activa").sum())
    pen = int((d.estado == "Pendiente").sum()); anu = int((d.estado == "Anulada").sum())
    ta = act / v if v else 0; tp = pen / v if v else 0; tan = anu / v if v else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VENDIDAS", f"{v:,}")
    c2.metric(f"ACTIVAS  {semaforo(ta, METAS['activacion'], 'maximo')}",
              f"{act:,}", f"{ta:.1%} (meta {METAS['activacion']:.0%})")
    c3.metric(f"PENDIENTES  {semaforo(tp, METAS['pendientes'], 'minimo')}",
              f"{pen:,}", f"{tp:.1%} (techo {METAS['pendientes']:.0%})")
    c4.metric(f"ANULADAS  {semaforo(tan, METAS['anulacion'], 'minimo')}",
              f"{anu:,}", f"{tan:.1%} (techo {METAS['anulacion']:.0%})")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Tablero", "🤖 Modelo predictivo", "🎯 Optimización",
         "🎲 Monte Carlo", "🗂️ Datos y evidencias"])

    # ---------------- TAB 1: Tablero ----------------
    with tab1:
        resumen = tabla_resumen(d)
        st.subheader("Tabla consolidada (KPIs · metas · semáforos · alertas)")
        st.dataframe(resumen, use_container_width=True, hide_index=True)
        g1, g2 = st.columns(2)
        rp = (d.groupby("agencia").estado.value_counts().unstack().fillna(0))
        fig1 = px.bar(rp, barmode="stack", title="Estados por agencia (filtrado)",
                      color_discrete_map={"Activa": "#1a9850", "Pendiente": "#fee08b",
                                          "Anulada": "#d73027"})
        fig1.add_hline(y=0)
        g1.plotly_chart(fig1, use_container_width=True)
        fig2 = px.funnel(d.estado.value_counts().rename_axis("Estado").reset_index(name="n"),
                         x="n", y="Estado", title="Embudo de estados",
                         color="Estado",
                         color_discrete_map={"Activa": "#1a9850", "Pendiente": "#fee08b",
                                             "Anulada": "#d73027"})
        g2.plotly_chart(fig2, use_container_width=True)
        diaria = (d.assign(fecha=d.fecha_venta.dt.date).groupby(["fecha", "estado"])
                    .size().unstack().fillna(0).reset_index())
        st.plotly_chart(px.line(diaria, x="fecha", y=[c for c in diaria.columns if c != "fecha"],
                                markers=True, title="Tendencia diaria por estado"),
                        use_container_width=True)
        st.info("Regla de alerta (umbral provisional): agencias con activación < "
                f"{METAS['umbral_alerta']:.0%} → revisión prioritaria de pendientes y causas.")

    # ---------------- TAB 2: Modelo predictivo ----------------
    with tab2:
        st.subheader("Clasificación: probabilidad de activación (regresión logística)")
        st.caption("Split temporal: entrenamiento ≤ 04-sep, prueba > 04-sep · sin fuga de información · "
                   "uso: ordenar seguimiento, NO sancionar colaboradores")
        model, met = modelo_predictivo(df)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AUC", f"{met['auc']:.3f}")
        m2.metric("Precisión", f"{met['precision']:.3f}")
        m3.metric("Recall", f"{met['recall']:.3f}")
        m4.metric("F1", f"{met['f1']:.3f}")
        st.text("Matriz de confusión [[TN, FP],[FN, TP]]: " + str(met["matriz_confusion"]))
        st.text(met["reporte"])
        st.subheader("Variables más influyentes (top 10 coeficientes)")
        st.dataframe(top_variables(model), use_container_width=True, hide_index=True)

    # ---------------- TAB 3: Optimización ----------------
    with tab3:
        st.subheader("Asignación óptima de contactos a pendientes (programación lineal)")
        st.caption(f"Supuestos documentados: costo S/ {SUPUESTOS['costo_contacto']} por contacto, "
                   f"presupuesto S/ {SUPUESTOS['presupuesto']}, conversión base "
                   f"{SUPUESTOS['conv_base_contactado']:.0%} ajustada por factor de agencia")
        res, out = optimizacion(df)
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.success(f"Estado del solver: {res.message} · Activaciones incrementales esperadas: "
                   f"{out.loc['TOTAL','activaciones_esperadas']} · "
                   f"Costo total: S/ {out.loc['TOTAL','costo_S']}")

    # ---------------- TAB 4: Monte Carlo ----------------
    with tab4:
        st.subheader("Simulación de Monte Carlo: activaciones incrementales del piloto")
        st.caption(f"{SUPUESTOS['n_sim']:,} iteraciones · respuesta ~ Beta(8,20) · conversión ~ Beta(12,28) "
                   f"· factores por agencia documentados")
        s = monte_carlo()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("P10", f"{s['P10']:.0f}")
        k2.metric("P50", f"{s['P50']:.0f}")
        k3.metric("P90", f"{s['P90']:.0f}")
        k4.metric(f"P(≥ meta {SUPUESTOS['meta_incremental']})", s["P(>=meta)"])
        st.write("**Regla de decisión:** escalar el piloto si P50 mejora la activación sin elevar "
                 "anulaciones ni exceder capacidad; documentar entradas y validar con expertos solo "
                 "como supuestos explícitos.")

    # ---------------- TAB 5: Datos y evidencias ----------------
    with tab5:
        st.subheader("Datos filtrados (nivel tarjeta) y registro colaborador-día")
        st.dataframe(d, use_container_width=True)
        st.dataframe(reg, use_container_width=True)
        st.download_button("⬇️ Descargar CSV tarjetas (filtrado)",
                           d.to_csv(index=False).encode(), "tarjetas_filtrado.csv")
        st.download_button("⬇️ Descargar tabla resumen con semáforos",
                           tabla_resumen(d).to_csv(index=False).encode(),
                           "tabla_resumen_semaforos.csv")
        st.download_button("⬇️ Descargar registros colaborador-día (175)",
                           reg.to_csv(index=False).encode(), "registros_175.csv")


# ---------------------------------------------------------------------------
# 7. MODO CONSOLA: evidencias reproducibles (sin Streamlit)
# ---------------------------------------------------------------------------
def exportar(outdir="evidencia_out"):
    os.makedirs(outdir, exist_ok=True)
    df, reg = generar_datos()

    # comprobación de coherencia con la línea base del documento
    chk = (df.groupby("agencia").estado.value_counts().unstack().fillna(0).astype(int))
    for ag, d in LINEA_BASE.items():
        assert chk.loc[ag, "Activa"] == d["activas"], ag
        assert chk.loc[ag, "Pendiente"] == d["pendientes"], ag
        assert chk.loc[ag, "Anulada"] == d["anuladas"], ag
    assert len(df) == 2620 and len(reg) == 175
    print("✔ Datos coherentes con la línea base: 2.620 tarjetas, 175 registros, "
          "estados por agencia exactos.")

    df.to_csv(f"{outdir}/01_tarjetas_nivel_tarjeta.csv", index=False)
    reg.to_csv(f"{outdir}/02_registros_colaborador_dia.csv", index=False)
    tabla_resumen(df).to_csv(f"{outdir}/03_tabla_resumen_semaforos.csv", index=False)

    model, met = modelo_predictivo(df)
    with open(f"{outdir}/04_metricas_modelo.txt", "w") as f:
        f.write(f"AUC={met['auc']:.4f}\nPrecision={met['precision']:.4f}\n"
                f"Recall={met['recall']:.4f}\nF1={met['f1']:.4f}\n"
                f"MatrizConfusion={met['matriz_confusion']}\n\n{met['reporte']}\n")
    top_variables(model).to_csv(f"{outdir}/05_variables_modelo.csv", index=False)

    res, out = optimizacion(df)
    out.to_csv(f"{outdir}/06_optimizacion_contactos.csv")
    monte_carlo().to_csv(f"{outdir}/07_montecarlo_resumen.csv")
    print(f"✔ Evidencias exportadas en ./{outdir}/ "
          f"(AUC={met['auc']:.3f}, F1={met['f1']:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true",
                        help="genera CSVs de evidencia sin Streamlit")
    args = parser.parse_args()
    if args.export:
        exportar()
    else:
        run_dashboard()
