"""Dashboard de monitoreo de fraude – PagoSeguro (Streamlit).

    API_URL=http://localhost:8000 API_KEY=demo-key streamlit run dashboard/app.py

Consume exclusivamente la API REST:
* Monitor: KPIs, decisiones en el tiempo, distribución de probabilidades, últimas transacciones.
* Explicaciones: reporte SHAP de cada rechazo / revisión (auditoría CMF).
* Probar transacción: envía una transacción y muestra decisión + explicación.
* Modelo: versión, umbrales y métricas del modelo desplegado.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.charts import (  # noqa: E402
    badge, decisions_over_time, probability_histogram, shap_contributions,
)
from dashboard.client import ApiError, FraudApiClient  # noqa: E402

PCA = [f"V{i}" for i in range(1, 29)]
# Filas reales del dataset (mlg-ulb) usadas cuando no hay models/sample_payloads.json
BUILTIN_SAMPLES = {
    "Legítima (ejemplo)": {"Time": 0.0, "Amount": 149.62, **dict(zip(PCA, [
        -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 0.239599, 0.098698, 0.363787,
        0.090794, -0.551600, -0.617801, -0.991390, -0.311169, 1.468177, -0.470401, 0.207971, 0.025791,
        0.403993, 0.251412, -0.018307, 0.277838, -0.110474, 0.066928, 0.128539, -0.189115, 0.133558,
        -0.021053]))},
    "Fraude (ejemplo)": {"Time": 406.0, "Amount": 0.0, **dict(zip(PCA, [
        -2.312227, 1.951992, -1.609851, 3.997906, -0.522188, -1.426545, -2.537387, 1.391657, -2.770089,
        -2.772272, 3.202033, -2.899907, -0.595222, -4.289254, 0.389724, -1.140747, -2.830056, -0.016822,
        0.416956, 0.126911, 0.517232, -0.035049, -0.465211, 0.320198, 0.044519, 0.177840, 0.261145,
        -0.143276]))},
}

st.set_page_config(page_title="PagoSeguro · Monitor de Fraude", page_icon="🛡️", layout="wide")


def load_samples() -> dict[str, dict]:
    path = Path(os.environ.get("SAMPLES_PATH", ROOT / "models" / "sample_payloads.json"))
    samples = dict(BUILTIN_SAMPLES)
    if path.exists():
        data = json.loads(path.read_text())
        for kind, label in (("legit", "Legítima"), ("fraud", "Fraude")):
            for i, tx in enumerate(data.get(kind, []), start=1):
                samples[f"{label} del set de test #{i}"] = tx
    return samples


# ───────────────────────────── Barra lateral ─────────────────────────────
with st.sidebar:
    st.title("🛡️ PagoSeguro")
    st.caption("Detección de fraude en tiempo real")
    api_url = st.text_input("URL de la API", os.environ.get("API_URL", "http://localhost:8000"))
    api_key = st.text_input("API key", os.environ.get("API_KEY", ""), type="password")
    hours = st.selectbox("Ventana de tiempo", [1, 6, 24, 24 * 7], index=2,
                         format_func=lambda h: f"Últimas {h} h" if h < 48 else f"Últimos {h // 24} días")
    refresh = st.toggle("Actualizar automáticamente", value=False)
    every = st.select_slider("Cada", options=[5, 10, 30, 60], value=10, format_func=lambda s: f"{s} s",
                             disabled=not refresh)

client = FraudApiClient(api_url, api_key or None)

with st.sidebar:
    st.divider()
    try:
        h = client.health()
        ok = h["status"] == "ok"
        st.markdown(f"**Estado:** {'🟢 operativo' if ok else '🟠 degradado'}")
        st.markdown(f"**Modelo:** `{h['model_version']}`  \n**Backend:** `{h['scorer']}`")
        icons = {"connected": "🟢 conectado", "disconnected": "🔴 desconectado", "disabled": "⚪ deshabilitado"}
        st.markdown(f"**RabbitMQ:** {icons[h['rabbitmq']]}  \n**Auditoría:** {icons[h['audit_store']]}")
    except ApiError as err:
        st.error(f"API no disponible: {err.detail}")
        st.stop()


def api_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ApiError as err:
        if err.status == 401:
            st.error("API key inválida: ingrésala en la barra lateral.")
        elif err.status == 503:
            st.warning(f"Servicio no disponible: {err.detail}")
        else:
            st.error(f"Error de la API ({err.status}): {err.detail}")
        return None


def fmt_money(x: float) -> str:
    return f"${x:,.0f}".replace(",", ".")


tab_monitor, tab_explain, tab_test, tab_model = st.tabs(
    ["📊 Monitor", "🔍 Explicaciones", "🧪 Probar transacción", "🧠 Modelo"])

# ───────────────────────────── Monitor ─────────────────────────────
with tab_monitor:
    @st.fragment(run_every=every if refresh else None)
    def monitor():
        model = api_call(client.model)
        stats = api_call(client.stats, hours)
        txs = api_call(client.transactions, limit=500, hours=hours) or []
        if not stats or not model:
            return
        by = stats["by_decision"]
        count = lambda d: by.get(d, {}).get("count", 0)  # noqa: E731
        total = stats["total"]
        c = st.columns(5)
        c[0].metric("Transacciones", f"{total:,}".replace(",", "."))
        c[1].metric(f"{badge('REJECT')} · {count('REJECT')} tx", f"{stats['reject_rate']:.2%}")
        c[2].metric(f"{badge('REVIEW')} · {count('REVIEW')} tx",
                    f"{count('REVIEW') / total:.2%}" if total else "–")
        c[3].metric("Monto bloqueado", fmt_money(by.get("REJECT", {}).get("amount", 0)))
        c[4].metric("SHAP promedio", f"{stats['avg_explain_ms']:.1f} ms" if stats["avg_explain_ms"] else "–")

        if not txs:
            st.info("Aún no hay transacciones auditadas en esta ventana. Envía una desde «Probar transacción».")
            return
        left, right = st.columns(2)
        fig, freq = decisions_over_time(txs)
        left.markdown(f"**Decisiones por intervalo de {freq}**")
        left.plotly_chart(fig, width="stretch")
        right.markdown("**Distribución de probabilidades de fraude**")
        right.plotly_chart(probability_histogram(txs, model["threshold"], model["review_threshold"]),
                           width="stretch")

        st.subheader("Últimas transacciones")
        df = pd.DataFrame(txs)
        df["Decisión"] = df["decision"].map(badge)
        df["scored_at"] = pd.to_datetime(df["scored_at"], format="ISO8601")
        df["card_masked"] = df["card_masked"].fillna("—")
        df["Razón principal"] = df["top_reasons"].map(
            lambda r: f"{r[0]['label']} ({r[0]['shap']:+.2f})" if isinstance(r, list) and r else "—")
        st.dataframe(
            df[["scored_at", "transaction_id", "Decisión", "fraud_probability", "amount", "card_masked",
                "Razón principal", "scorer"]].head(100),
            hide_index=True, width="stretch",
            column_config={
                "scored_at": st.column_config.DatetimeColumn("Fecha", format="DD/MM HH:mm:ss"),
                "transaction_id": "ID",
                "fraud_probability": st.column_config.ProgressColumn(
                    "Prob. fraude", min_value=0.0, max_value=1.0, format="%.3f"),
                "amount": st.column_config.NumberColumn("Monto", format="$%.2f"),
                "card_masked": "Tarjeta",
                "scorer": "Backend",
            })
        st.caption(f"Actualizado {datetime.now():%H:%M:%S} · {len(txs)} transacciones en la ventana")

    monitor()

# ───────────────────────────── Explicaciones ─────────────────────────────
def show_explanation(exp: dict, key: str, header: bool = True) -> None:
    """``key`` distingue el gráfico cuando el mismo reporte aparece en dos pestañas."""
    if header:
        c = st.columns(4)
        c[0].metric("Decisión", badge(exp["decision"]))
        c[1].metric("Probabilidad de fraude", f"{exp['fraud_probability']:.2%}")
        c[2].metric("Umbral de rechazo", f"{exp['threshold']:.2%}")
        c[3].metric("Monto", f"${exp['amount']:,.2f}" if exp.get("amount") is not None else "–")
    if exp.get("explanation_text"):
        st.info(exp["explanation_text"], icon="🧾")
    if exp.get("shap_values"):
        st.markdown("**Principales factores de la decisión**")
        st.plotly_chart(shap_contributions(exp), width="stretch", key=f"shap-{key}")
        st.caption(f"Valor base del modelo (log-odds): {exp['shap_base_value']:.3f}. "
                   "Base + Σ contribuciones = log-odds de la probabilidad. "
                   f"Modelo `{exp['model_version']}` · SHAP en {exp['explain_ms']:.1f} ms.")
        with st.expander("Tabla completa de contribuciones y variables", key=f"table-{key}"):
            table = pd.DataFrame({"SHAP": exp["shap_values"]})
            table["Valor"] = [exp["features"].get(f) for f in table.index]
            st.dataframe(table.sort_values("SHAP", key=abs, ascending=False), width="stretch")
    else:
        st.caption("Transacción aprobada: se registra sin SHAP (EXPLAIN_DECISIONS=REJECT,REVIEW).")


with tab_explain:
    st.subheader("Reporte de explicabilidad por transacción")
    flagged = api_call(client.transactions, limit=100, hours=hours) or []
    flagged = [t for t in flagged if t["decision"] in ("REJECT", "REVIEW")]
    col_a, col_b = st.columns([2, 1])
    options = {f"{badge(t['decision'])} · {t['transaction_id']} · {t['fraud_probability']:.1%}": t["transaction_id"]
               for t in flagged}
    choice = col_a.selectbox("Rechazos y revisiones recientes", list(options) or ["(sin rechazos en la ventana)"])
    manual = col_b.text_input("…o busca por ID de transacción")
    tx_id = manual.strip() or options.get(choice)
    if tx_id:
        exp = api_call(client.explanation, tx_id)
        if exp is None:
            st.warning(f"No hay reporte para `{tx_id}` (inexistente o aún en cola del worker).")
        else:
            show_explanation(exp, key="explain-tab")

# ───────────────────────────── Probar transacción ─────────────────────────────
with tab_test:
    st.subheader("Enviar una transacción a la API")
    samples = load_samples()
    base_name = st.selectbox("Partir de", list(samples))
    base = samples[base_name]
    with st.form("score"):
        c = st.columns(3)
        amount = c[0].number_input("Monto", min_value=0.0, value=float(base["Amount"]), step=1.0)
        hour = c[1].number_input("Hora del día", 0, 23, int((base["Time"] % 86400) // 3600))
        card = c[2].text_input("Número de tarjeta (opcional, se enmascara)", placeholder="4111 1111 1111 1234")
        pca_text = st.text_area("Componentes PCA V1..V28 (JSON)",
                                json.dumps({k: base[k] for k in PCA}, indent=None), height=110)
        submitted = st.form_submit_button("Evaluar transacción", type="primary")
    if submitted:
        try:
            pca = json.loads(pca_text)
        except json.JSONDecodeError:
            st.error("El JSON de componentes PCA no es válido.")
            st.stop()
        tx = {**pca, "Amount": amount, "Time": float(hour * 3600),
              "transaction_id": f"dash-{int(time.time() * 1000)}"}
        if card.strip():
            tx["card_number"] = card
        t0 = time.perf_counter()
        res = api_call(client.score, tx)
        rtt = (time.perf_counter() - t0) * 1000
        if res:
            c = st.columns(4)
            c[0].metric("Decisión", badge(res["decision"]))
            c[1].metric("Probabilidad de fraude", f"{res['fraud_probability']:.2%}")
            c[2].metric("Latencia API", f"{res['latency_ms']:.1f} ms",
                        help=f"Tiempo de procesamiento en la API. Ida y vuelta desde el dashboard: {rtt:.0f} ms")
            c[3].metric("Backend", res["scorer"])
            with st.expander("Respuesta JSON de la API"):
                st.json(res)
            if res["explanation_status"] == "queued" and res["decision"] != "APPROVE":
                with st.spinner("Esperando el reporte SHAP del worker…"):
                    exp = None
                    for _ in range(25):
                        exp = client.explanation(res["transaction_id"])
                        if exp:
                            break
                        time.sleep(0.2)
                if exp:
                    show_explanation(exp, key="test-tab", header=False)
                else:
                    st.warning("El worker aún no publica el reporte; revísalo luego en «Explicaciones».")
            elif res["explanation_status"] == "unavailable":
                st.warning("RabbitMQ no disponible: la decisión se emitió, pero no se generará reporte SHAP.")

# ───────────────────────────── Modelo ─────────────────────────────
with tab_model:
    model = api_call(client.model)
    if model:
        st.subheader(f"Modelo `{model['model_version']}`")
        c = st.columns(4)
        c[0].metric("Umbral de rechazo", f"{model['threshold']:.3f}")
        c[1].metric("Umbral de revisión", f"{model['review_threshold']:.3f}")
        c[2].metric("Backend", model["scorer_backend"])
        c[3].metric("Endpoint SageMaker", model.get("sagemaker_endpoint") or "—")
        st.caption(f"Política de umbral: `{model.get('threshold_policy')}` · entrenado {model.get('trained_at')}")
        m = model.get("metrics_test", {})
        if m:
            names = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1-score",
                     "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}
            st.markdown("**Métricas en el conjunto de test**")
            mc = st.columns(len(names))
            for col, (k, label) in zip(mc, names.items()):
                if k in m:
                    col.metric(label, f"{m[k]:.4f}")
            fp, loss = m.get("false_positive_share_of_blocks"), m.get("residual_fraud_loss_%_of_volume")
            if fp is not None and loss is not None:
                st.markdown("**Objetivos de negocio**")
                rows = [
                    {"Objetivo": "Falsos positivos sobre bloqueos", "Meta": "< 5 %", "Resultado": f"{fp:.2%}",
                     "Cumple": "✅ Sí" if fp < 0.05 else "❌ No"},
                    {"Objetivo": "Pérdida por fraude, solo bloqueos", "Meta": "< 0,1 %", "Resultado": f"{loss:.3%}",
                     "Cumple": "✅ Sí" if loss < 0.001 else "❌ No"},
                ]
                loss_rev = m.get("residual_fraud_loss_with_review_%_of_volume")
                if loss_rev is not None:
                    rows.append({"Objetivo": "Pérdida por fraude con zona de revisión", "Meta": "< 0,1 %",
                                 "Resultado": f"{loss_rev:.3%}", "Cumple": "✅ Sí" if loss_rev < 0.001 else "❌ No"})
                    rows.append({"Objetivo": "Clientes legítimos enviados a verificación", "Meta": "—",
                                 "Resultado": f"{m.get('legit_sent_to_review_%', 0):.3%}", "Cumple": "costo"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with st.expander(f"Variables del modelo ({len(model['feature_columns'])})"):
            st.write(", ".join(model["feature_columns"]))
