"""
App de Streamlit: simulación Monte Carlo de la concentración de un mercado (HHI).
Ejecutar en local:  streamlit run app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ---------------------------------------------------------------
# 1. FUNCIONES DE CÁLCULO (iguales que antes)
# ---------------------------------------------------------------
def calcular_hhi(cuotas):
    cuotas = np.asarray(cuotas, dtype=float)
    return np.sum(cuotas ** 2, axis=-1)


def calcular_cr(cuotas, k=4):
    cuotas = np.asarray(cuotas, dtype=float)
    ordenadas = -np.sort(-cuotas, axis=-1)
    return ordenadas[..., :k].sum(axis=-1)


def clasificar_hhi(hhi):
    if hhi < 1000:
        return "No concentrado"
    elif hhi <= 1800:
        return "Moderadamente concentrado"
    return "Altamente concentrado"


def simular_mercados(n_empresas, n_sim, alpha, semilla):
    rng = np.random.default_rng(semilla)
    return rng.dirichlet(np.full(n_empresas, alpha), size=n_sim) * 100


# ---------------------------------------------------------------
# 2. INTERFAZ: TÍTULO Y CONTROLES
# ---------------------------------------------------------------
st.title("Concentración de mercado: simulación Monte Carlo")
st.write("Compara el HHI de un mercado real con miles de mercados aleatorios "
         "con el mismo número de empresas.")

texto_cuotas = st.text_input(
    "Cuotas de mercado en % (separadas por comas, deben sumar 100)",
    value="40, 25, 15, 12, 8",
)

st.sidebar.header("Parámetros de la simulación")
n_sim = st.sidebar.slider("Número de simulaciones", 1_000, 50_000, 10_000, step=1_000)
alpha = st.sidebar.slider("Alpha (desigualdad de los mercados simulados)", 0.1, 10.0, 1.0, step=0.1)
semilla = st.sidebar.number_input("Semilla", value=42, step=1)


# ---------------------------------------------------------------
# 3. LEER Y VALIDAR LAS CUOTAS
# ---------------------------------------------------------------
try:
    cuotas_reales = np.array([float(x) for x in texto_cuotas.split(",") if x.strip()])
except ValueError:
    st.error("Escribe solo números separados por comas, por ejemplo: 40, 25, 15, 12, 8")
    st.stop()

if len(cuotas_reales) < 2:
    st.error("Introduce al menos 2 empresas.")
    st.stop()

if not np.isclose(cuotas_reales.sum(), 100):
    st.error(f"Las cuotas deben sumar 100 (ahora suman {cuotas_reales.sum():.2f}).")
    st.stop()


# ---------------------------------------------------------------
# 4. SIMULAR Y CALCULAR
# ---------------------------------------------------------------
n = len(cuotas_reales)
simulados = simular_mercados(n, n_sim, alpha, int(semilla))
hhi_sim = calcular_hhi(simulados)

hhi_real = float(calcular_hhi(cuotas_reales))
cr4_real = float(calcular_cr(cuotas_reales, k=4))
percentil = float(np.mean(hhi_sim <= hhi_real) * 100)
p025, p975 = np.percentile(hhi_sim, [2.5, 97.5])


# ---------------------------------------------------------------
# 5. MOSTRAR RESULTADOS EN LA PÁGINA
# ---------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("HHI real", f"{hhi_real:,.0f}")
col2.metric("CR4 real", f"{cr4_real:.1f} %")
col3.metric("Percentil", f"{percentil:.1f} %")

st.write(f"**Clasificación:** {clasificar_hhi(hhi_real)}")
st.write(f"El caso real es más concentrado que el **{percentil:.1f} %** de los "
         f"{n_sim:,} mercados simulados con {n} empresas.")
st.write(f"HHI simulado: media {hhi_sim.mean():,.0f}, "
         f"intervalo 95 % entre {p025:,.0f} y {p975:,.0f}. "
         f"(Con {n} empresas el HHI mínimo posible es {10000 / n:,.0f}.)")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(hhi_sim, bins=60, color="steelblue", alpha=0.7, label="Mercados simulados")
ax.axvline(hhi_real, color="crimson", linewidth=2, label=f"Caso real (HHI = {hhi_real:.0f})")
ax.set_xlabel("Índice HHI")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución Monte Carlo del HHI")
ax.legend()
st.pyplot(fig)
