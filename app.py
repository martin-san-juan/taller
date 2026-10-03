"""
App de Streamlit: simulación Monte Carlo de la concentración de un mercado.
Indicadores: HHI, CR_k, índice de dominancia, entropía y entropía normalizada.
Ejecutar en local:  streamlit run app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ---------------------------------------------------------------
# 1. INDICADORES DE CONCENTRACIÓN
#    Todas las funciones reciben cuotas en % (que suman 100).
#    Funcionan con un solo mercado (lista) o con muchos a la vez (matriz).
# ---------------------------------------------------------------
def calcular_hhi(cuotas):
    """HHI = suma de los cuadrados de las cuotas. Rango: 10.000/N a 10.000."""
    cuotas = np.asarray(cuotas, dtype=float)
    return np.sum(cuotas ** 2, axis=-1)


def calcular_cr(cuotas, k):
    """CR_k = suma de las cuotas de las k empresas más grandes."""
    cuotas = np.asarray(cuotas, dtype=float)
    ordenadas = -np.sort(-cuotas, axis=-1)          # de mayor a menor
    return ordenadas[..., :k].sum(axis=-1)


def calcular_dominancia(cuotas):
    """Índice de dominancia (García Alba): ID = 10.000 * suma de (s_i² / HHI)².
    Mide cuánto del HHI se debe a la empresa más grande. Rango: 10.000/N a 10.000."""
    cuotas = np.asarray(cuotas, dtype=float)
    hhi = np.expand_dims(calcular_hhi(cuotas), -1)  # añade una dimensión para dividir fila a fila
    aportes = cuotas ** 2 / hhi                     # parte del HHI que aporta cada empresa
    return 10_000 * np.sum(aportes ** 2, axis=-1)


def calcular_entropia(cuotas):
    """Entropía = -suma de p_i * ln(p_i), con p_i en proporción (0 a 1).
    Ojo: MÁS entropía = MENOS concentración. Rango: 0 a ln N."""
    p = np.asarray(cuotas, dtype=float) / 100
    p_segura = np.where(p > 0, p, 1.0)              # evita ln(0); como 0*ln(0)=0, se usa ln(1)=0
    return -np.sum(p * np.log(p_segura), axis=-1)


def calcular_entropia_normalizada(cuotas):
    """Entropía dividida por ln N. Rango: 0 (monopolio) a 1 (todas iguales)."""
    n = np.asarray(cuotas).shape[-1]
    return calcular_entropia(cuotas) / np.log(n)


# Información de cada indicador: si un valor alto significa más concentración,
# cómo mostrar los números y una breve descripción.
INDICADORES = {
    "HHI (Herfindahl-Hirschman)": {
        "mas_alto_mas_concentrado": True,
        "formato": "{:,.0f}",
        "descripcion": "Suma de los cuadrados de las cuotas. Va de 10.000/N a 10.000 (monopolio).",
    },
    "CR_k (cuota de las k mayores)": {
        "mas_alto_mas_concentrado": True,
        "formato": "{:.1f} %",
        "descripcion": "Porcentaje del mercado en manos de las k empresas más grandes.",
    },
    "Índice de dominancia (ID)": {
        "mas_alto_mas_concentrado": True,
        "formato": "{:,.0f}",
        "descripcion": "Mide si el HHI se debe sobre todo a una empresa. "
                       "Va de 10.000/N (todas iguales) a 10.000.",
    },
    "Entropía": {
        "mas_alto_mas_concentrado": False,
        "formato": "{:.3f}",
        "descripcion": "Va de 0 (monopolio) a ln N (todas iguales). "
                       "Valores MÁS ALTOS indican MENOS concentración.",
    },
    "Entropía normalizada (E / ln N)": {
        "mas_alto_mas_concentrado": False,
        "formato": "{:.3f}",
        "descripcion": "Entropía dividida por ln N, para comparar mercados con distinto "
                       "número de empresas. Va de 0 a 1; más alto = menos concentración.",
    },
}


def calcular_indicador(nombre, cuotas, k=4):
    """Llama a la función que corresponde al indicador elegido."""
    if nombre.startswith("HHI"):
        return calcular_hhi(cuotas)
    if nombre.startswith("CR"):
        return calcular_cr(cuotas, k)
    if nombre.startswith("Índice de dominancia"):
        return calcular_dominancia(cuotas)
    if nombre.startswith("Entropía normalizada"):
        return calcular_entropia_normalizada(cuotas)
    return calcular_entropia(cuotas)


def clasificar_hhi(hhi):
    """Umbrales de las Merger Guidelines de EE.UU. (2023)."""
    if hhi < 1000:
        return "No concentrado"
    elif hhi <= 1800:
        return "Moderadamente concentrado"
    return "Altamente concentrado"


# ---------------------------------------------------------------
# 2. SIMULACIÓN MONTE CARLO
#    @st.cache_data guarda el resultado: si solo cambias de indicador,
#    no se vuelven a simular los mercados (la app va más rápido).
# ---------------------------------------------------------------
@st.cache_data
def simular_mercados(n_empresas, n_sim, alpha, semilla):
    rng = np.random.default_rng(semilla)
    return rng.dirichlet(np.full(n_empresas, alpha), size=n_sim) * 100


# ---------------------------------------------------------------
# 3. INTERFAZ: ENTRADA DE DATOS
# ---------------------------------------------------------------
st.title("Concentración de mercado: simulación Monte Carlo")
st.write("Compara un mercado real con miles de mercados aleatorios "
         "con el mismo número de empresas.")

texto_cuotas = st.text_input(
    "Cuotas de mercado en % (separadas por comas, deben sumar 100)",
    value="40, 25, 15, 12, 8",
)


# ---------------------------------------------------------------
# 4. VALIDACIÓN DE LAS CUOTAS
# ---------------------------------------------------------------
try:
    cuotas_reales = np.array([float(x) for x in texto_cuotas.split(",") if x.strip()])
except ValueError:
    st.error("Escribe solo números separados por comas, por ejemplo: 40, 25, 15, 12, 8")
    st.stop()

if len(cuotas_reales) < 2:
    st.error("Introduce al menos 2 empresas.")
    st.stop()

fuera_de_rango = [x for x in cuotas_reales if x < 0 or x > 100]
if fuera_de_rango:
    lista = ", ".join(f"{x:g}" for x in fuera_de_rango)
    st.error(f"Cada cuota debe estar entre 0 y 100. Valores no válidos: {lista}")
    st.stop()

if not np.isclose(cuotas_reales.sum(), 100):
    st.error(f"Las cuotas deben sumar 100 (ahora suman {cuotas_reales.sum():.2f}).")
    st.stop()

n = len(cuotas_reales)


# ---------------------------------------------------------------
# 5. BARRA LATERAL: ELECCIÓN DE INDICADOR Y PARÁMETROS
# ---------------------------------------------------------------
st.sidebar.header("Indicador")
indicador = st.sidebar.selectbox("¿Qué indicador quieres analizar?", list(INDICADORES.keys()))

k = min(4, n)
if indicador.startswith("CR"):
    # El máximo de k es el número de empresas, así nunca se puede superar
    k = st.sidebar.slider("k (número de empresas más grandes)", 1, n, min(4, n))

st.sidebar.header("Simulación")
n_sim = int(st.sidebar.number_input(
    "Número de simulaciones",
    min_value=1,          # permite valores pequeños (menos de 1.000)
    max_value=500_000,    # tope de seguridad para no saturar el servidor
    value=1_000,          # valor por defecto
    step=100,             # cuánto suben/bajan los botones + y -
))
if n_sim > 1_000:
    st.sidebar.warning(
        f"Vas a simular {n_sim:,} mercados. Más simulaciones dan resultados más "
        "precisos, pero la app tardará más en responder y usará más memoria y CPU "
        "del servidor. En Streamlit Cloud los recursos son limitados y una cifra "
        "muy alta puede hacer que la app se ralentice o se reinicie."
    )
alpha = 1.0   # fijo: todos los repartos posibles del mercado son igual de probables
semilla = 42  # fija: así los resultados son siempre los mismos para los mismos datos


# ---------------------------------------------------------------
# 6. RESUMEN DEL CASO REAL CON TODOS LOS INDICADORES
# ---------------------------------------------------------------
st.subheader("Caso real: todos los indicadores")
filas = []
for nombre, datos in INDICADORES.items():
    valor = float(calcular_indicador(nombre, cuotas_reales, k))
    etiqueta = f"CR{k} (cuota de las {k} mayores)" if nombre.startswith("CR") else nombre
    filas.append({"Indicador": etiqueta, "Valor": datos["formato"].format(valor)})
st.table(filas)


# ---------------------------------------------------------------
# 7. ANÁLISIS MONTE CARLO DEL INDICADOR ELEGIDO
# ---------------------------------------------------------------
info = INDICADORES[indicador]
nombre_corto = f"CR{k}" if indicador.startswith("CR") else indicador

simulados = simular_mercados(n, n_sim, alpha, int(semilla))
valores_sim = calcular_indicador(indicador, simulados, k)
valor_real = float(calcular_indicador(indicador, cuotas_reales, k))

# Percentil: % de mercados simulados con un valor menor o igual al real
percentil = float(np.mean(valores_sim <= valor_real) * 100)

# % de mercados simulados MENOS concentrados que el real
# (en la entropía la dirección se invierte: menos concentrado = valor más alto)
if info["mas_alto_mas_concentrado"]:
    mas_concentrado_que = float(np.mean(valores_sim < valor_real) * 100)
else:
    mas_concentrado_que = float(np.mean(valores_sim > valor_real) * 100)

p025, p975 = np.percentile(valores_sim, [2.5, 97.5])
fmt = info["formato"]

st.subheader(f"Análisis Monte Carlo: {nombre_corto}")
st.caption(info["descripcion"])

col1, col2, col3 = st.columns(3)
col1.metric(f"{nombre_corto} real", fmt.format(valor_real))
col2.metric("Media simulada", fmt.format(float(valores_sim.mean())))
col3.metric("Percentil del caso real", f"{percentil:.1f} %")

st.write(f"El caso real es **más concentrado que el {mas_concentrado_que:.1f} %** de los "
         f"{n_sim:,} mercados simulados con {n} empresas.")
st.write(f"El 95 % de los mercados simulados tiene un {nombre_corto} entre "
         f"{fmt.format(float(p025))} y {fmt.format(float(p975))}.")

if indicador.startswith("CR") and k == n:
    st.info(f"Con k = {n} (todas las empresas), CR siempre vale 100 %, "
            "así que la comparación no aporta información. Prueba con un k menor.")

# Histograma del indicador elegido con la línea del caso real
# Si todos los valores simulados son iguales (CR con k = N), se usa una sola barra
if np.ptp(valores_sim) < 1e-6:
    barras, rango = 1, (valor_real - 1, valor_real + 1)
else:
    barras, rango = 60, None

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(valores_sim, bins=barras, range=rango, color="steelblue", alpha=0.7,
        label="Mercados simulados")
ax.axvline(valor_real, color="crimson", linewidth=2,
           label=f"Caso real ({fmt.format(valor_real)})")
ax.set_xlabel(nombre_corto)
ax.set_ylabel("Frecuencia")
ax.set_title(f"Distribución Monte Carlo: {nombre_corto}")
ax.legend()
st.pyplot(fig)
plt.close(fig)


# ---------------------------------------------------------------
# 8. PREGUNTA PARA EL USUARIO (según el HHI)
# ---------------------------------------------------------------
st.divider()
st.subheader("Pon a prueba tu intuición")

hhi_quiz = float(calcular_hhi(cuotas_reales))

# Traducimos la clasificación oficial a las tres opciones de la pregunta
RESPUESTA_CORRECTA = {
    "No concentrado": "Baja",
    "Moderadamente concentrado": "Moderada",
    "Altamente concentrado": "Alta",
}[clasificar_hhi(hhi_quiz)]

eleccion = st.radio(
    "Según el HHI, ¿la concentración de este mercado es baja, moderada o alta?",
    ["Baja", "Moderada", "Alta"],
    index=None,          # ninguna opción marcada al inicio
    horizontal=True,
)

if st.button("Responder"):
    if eleccion is None:
        st.warning("Elige una opción antes de responder.")
    elif eleccion == RESPUESTA_CORRECTA:
        st.success(f"¡Correcto! Con un HHI de {hhi_quiz:,.0f}, la concentración es "
                   f"{RESPUESTA_CORRECTA.lower()}.")
    else:
        st.error(f"No es correcto. Con un HHI de {hhi_quiz:,.0f}, la concentración es "
                 f"{RESPUESTA_CORRECTA.lower()}.")

    if eleccion is not None:
        st.caption("Criterio usado (Merger Guidelines de EE.UU., 2023): HHI menor que "
                   "1.000 = baja; entre 1.000 y 1.800 = moderada; mayor que 1.800 = alta.")
