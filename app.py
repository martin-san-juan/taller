 """
App de Streamlit: simulación Monte Carlo de la concentración de un mercado.
Indicadores: CR_k, HHI, índice de dominancia y entropía.
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


# Información de cada indicador: si un valor alto significa más concentración,
# cómo mostrar los números y una breve descripción.
INDICADORES = {
    "CR_k (ratio de concentración)": {
        "mas_alto_mas_concentrado": True,
        "formato": "{:.1f} %",
        "descripcion": "Porcentaje del mercado en manos de las k empresas más grandes.",
        "unidad": "% del mercado",
    },
    "HHI (Herfindahl-Hirschman)": {
        "mas_alto_mas_concentrado": True,
        "formato": "{:,.0f}",
        "descripcion": "Suma de los cuadrados de las cuotas. Va de 10.000/N a 10.000 (monopolio).",
        "unidad": "puntos, escala 0–10.000",
    },
    "Índice de dominancia (ID)": {
        "mas_alto_mas_concentrado": True,
        "formato": "{:,.0f}",
        "descripcion": "Mide si el HHI se debe sobre todo a una empresa. "
                       "Va de 10.000/N (todas iguales) a 10.000.",
        "unidad": "puntos, escala 0–10.000",
    },
    "Índice de entropía": {
        "mas_alto_mas_concentrado": False,
        "formato": "{:.3f}",
        "descripcion": "Va de 0 (monopolio) a ln N (todas iguales). "
                       "Valores MÁS ALTOS indican MENOS concentración.",
        "unidad": "nats, escala 0–ln N; más alto = menos concentrado",
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
    return calcular_entropia(cuotas)


# ---------------------------------------------------------------
# UMBRALES DE CLASIFICACIÓN (baja / moderada / alta)
# El HHI tiene umbrales oficiales (EE.UU. 2023: 1.000 y 1.800).
# Para los demás se usa la "equivalencia en número de empresas":
#   HHI 1.000 = mercado de 10 empresas iguales
#   HHI 1.800 = mercado de 10.000/1.800 ≈ 5,56 empresas iguales
# y se calcula cuánto valdría cada indicador en esos dos mercados.
# ---------------------------------------------------------------
N_EQ_BAJA = 10_000 / 1_000    # 10 empresas iguales
N_EQ_ALTA = 10_000 / 1_800    # ≈ 5,56 empresas iguales


def obtener_umbrales(nombre, n, k):
    """Devuelve (corte_baja, corte_alta, fuente) para el indicador elegido."""
    if nombre.startswith("HHI"):
        return 1_000, 1_800, "Fuente: umbrales oficiales de las Merger Guidelines de EE.UU. (2023)."
    if nombre.startswith("Índice de dominancia"):
        return 1_000, 1_800, ("Fuente: no hay umbrales oficiales vigentes. Con empresas iguales "
                              "el ID vale lo mismo que el HHI, así que se usan sus mismos cortes.")
    if nombre.startswith("CR"):
        baja = min(100.0, 100 * k / N_EQ_BAJA)
        alta = min(100.0, 100 * k / N_EQ_ALTA)
        return baja, alta, (f"Fuente: equivalencia con el HHI. Es el CR{k} de un mercado de 10 "
                            "empresas iguales (HHI 1.000) y de 5,56 empresas iguales (HHI 1.800).")
    return (np.log(N_EQ_BAJA), np.log(N_EQ_ALTA),
            "Fuente: equivalencia con el HHI. Con N empresas iguales la entropía vale ln N, "
            "así que los cortes son ln 10 y ln 5,56.")


def clasificar(nombre, valor, n, k, tol=1e-9):
    """Devuelve "Baja", "Moderada" o "Alta". En la entropía la escala va al revés."""
    baja, alta, _ = obtener_umbrales(nombre, n, k)
    if INDICADORES[nombre]["mas_alto_mas_concentrado"]:
        if valor < baja - tol:
            return "Baja"
        if valor <= alta + tol:
            return "Moderada"
        return "Alta"
    # Entropía: valor alto = poca concentración
    if valor > baja + tol:
        return "Baja"
    if valor >= alta - tol:
        return "Moderada"
    return "Alta"


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
def texto_desde_cuotas(cuotas):
    """Convierte una lista de cuotas en texto: [40, 25.5] -> "40, 25.5"."""
    return ", ".join(f"{c:.2f}".rstrip("0").rstrip(".") for c in cuotas)


def ajustar_cuotas_a_n():
    """Se ejecuta al cambiar N. Adapta las cuotas escritas al nuevo número de empresas
    manteniendo su forma, y las reescala para que vuelvan a sumar 100."""
    n_nuevo = st.session_state["n_empresas"]
    try:
        actuales = [float(x) for x in st.session_state["texto_cuotas"].split(",") if x.strip()]
    except ValueError:
        actuales = []
    actuales = [c for c in actuales if c > 0] or [100.0]   # si no hay nada válido, se parte de cero

    if len(actuales) >= n_nuevo:
        # Sobran empresas: se quedan las N primeras
        nuevas = actuales[:n_nuevo]
    else:
        # Faltan empresas: las nuevas reciben la cuota de la más pequeña
        nuevas = actuales + [min(actuales)] * (n_nuevo - len(actuales))

    # Reescalar para que sumen 100 y redondear a 2 decimales
    total = sum(nuevas)
    nuevas = [round(c * 100 / total, 2) for c in nuevas]
    nuevas[0] = round(nuevas[0] + 100 - sum(nuevas), 2)     # corrige el error de redondeo
    st.session_state["texto_cuotas"] = texto_desde_cuotas(nuevas)


def generar_caso_aleatorio():
    """Se ejecuta al pulsar el botón. Crea N cuotas al azar que suman 100.
    No usa semilla, así que cada clic da un mercado distinto. El resultado se guarda
    en session_state, por eso no cambia al elegir otro indicador."""
    n_actual = st.session_state["n_empresas"]
    rng = np.random.default_rng()                        # sin semilla: azar distinto cada vez
    cuotas = rng.dirichlet(np.ones(n_actual)) * 100
    cuotas = sorted((round(c, 2) for c in cuotas), reverse=True)   # de mayor a menor
    cuotas[0] = round(cuotas[0] + 100 - sum(cuotas), 2)  # corrige el error de redondeo
    st.session_state["texto_cuotas"] = texto_desde_cuotas(cuotas)


# Valores iniciales (solo la primera vez que se abre la página)
if "texto_cuotas" not in st.session_state:
    st.session_state["texto_cuotas"] = "40, 25, 15, 12, 8"
    st.session_state["n_empresas"] = 5

st.title("Concentración de mercado: simulación Monte Carlo")
st.write("Compara un mercado real con miles de mercados aleatorios "
         "con el mismo número de empresas.")

n = int(st.number_input(
    "Número de empresas (N)",
    min_value=2,
    max_value=100,
    step=1,
    key="n_empresas",
    on_change=ajustar_cuotas_a_n,   # al cambiar N, se ajustan las cuotas
))

texto_cuotas = st.text_input(
    f"Cuotas de mercado de las {n} empresas en % (separadas por comas, deben sumar 100)",
    key="texto_cuotas",
)
st.button("🎲 Generar caso real al azar", on_click=generar_caso_aleatorio)
st.caption("Al cambiar N, las cuotas se ajustan solas: si bajas N se quitan las últimas "
           "empresas, si lo subes se añaden empresas del tamaño de la más pequeña, y "
           "luego todo se reescala para sumar 100. Después puedes editarlas a mano.")


# ---------------------------------------------------------------
# 4. VALIDACIÓN DE LAS CUOTAS
# ---------------------------------------------------------------
try:
    cuotas_reales = np.array([float(x) for x in texto_cuotas.split(",") if x.strip()])
except ValueError:
    st.error("Escribe solo números separados por comas, por ejemplo: 40, 25, 15, 12, 8")
    st.stop()

if len(cuotas_reales) != n:
    st.error(f"Has escrito {len(cuotas_reales)} cuotas, pero N = {n}. "
             "Escribe una cuota por empresa o cambia N.")
    st.stop()

fuera_de_rango = [x for x in cuotas_reales if x < 0 or x > 100]
if fuera_de_rango:
    lista = ", ".join(f"{x:g}" for x in fuera_de_rango)
    st.error(f"Cada cuota debe estar entre 0 y 100. Valores no válidos: {lista}")
    st.stop()

if not np.isclose(cuotas_reales.sum(), 100):
    st.error(f"Las cuotas deben sumar 100 (ahora suman {cuotas_reales.sum():.2f}).")
    st.stop()


# ---------------------------------------------------------------
# 5. BARRA LATERAL: ELECCIÓN DE INDICADOR Y PARÁMETROS
# ---------------------------------------------------------------
st.sidebar.header("Indicador")
indicador = st.sidebar.selectbox("¿Qué indicador quieres analizar?", list(INDICADORES.keys()))

k = min(4, n)
if indicador.startswith("CR"):
    # El máximo de k es el número de empresas, así nunca se puede superar
    k = int(st.sidebar.number_input(
        "k (número de empresas más grandes)",
        min_value=1,
        max_value=n,          # nunca puede superar el número de empresas
        value=min(4, n),      # por defecto CR4 (o N si hay menos de 4 empresas)
        step=1,
    ))

st.sidebar.header("Simulación")
n_sim = int(st.sidebar.number_input(
    "Número de simulaciones",
    min_value=100,        # mínimo para que el percentil tenga sentido
    max_value=50_000,     # tope para no saturar el servidor
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
    if nombre.startswith("CR"):
        etiqueta = "CR1 (cuota de la mayor)" if k == 1 else f"CR{k} (cuota de las {k} mayores)"
    else:
        etiqueta = nombre
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
ax.set_xlabel(f"{nombre_corto} ({info['unidad']})")
ax.set_ylabel("Frecuencia (n.º de mercados simulados)")
ax.set_title(f"Distribución Monte Carlo: {nombre_corto}")
ax.legend()
st.pyplot(fig)
plt.close(fig)


# ---------------------------------------------------------------
# 8. PREGUNTA PARA EL USUARIO (según el indicador elegido)
# ---------------------------------------------------------------
st.divider()
st.subheader("Pon a prueba tu intuición")

correcta = clasificar(indicador, valor_real, n, k)
corte_baja, corte_alta, fuente = obtener_umbrales(indicador, n, k)

eleccion = st.radio(
    f"Según el indicador {nombre_corto}, ¿la concentración de este mercado es baja, moderada o alta?",
    ["Baja", "Moderada", "Alta"],
    index=None,                          # ninguna opción marcada al inicio
    horizontal=True,
    key=f"pregunta_{indicador}_{k}",     # se reinicia al cambiar de indicador o de k
)

if st.button("Responder"):
    if eleccion is None:
        st.warning("Elige una opción antes de responder.")
    else:
        texto = (f"Con {nombre_corto} = {fmt.format(valor_real)}, "
                 f"la concentración es {correcta.lower()}.")
        if eleccion == correcta:
            st.success("¡Correcto! " + texto)
        else:
            st.error("No es correcto. " + texto)

        # Explicar el criterio usado
        b, a = fmt.format(float(corte_baja)), fmt.format(float(corte_alta))
        if info["mas_alto_mas_concentrado"]:
            criterio = f"baja si es menor que {b}; moderada entre {b} y {a}; alta si es mayor que {a}."
        else:
            criterio = (f"baja si es mayor que {b}; moderada entre {a} y {b}; alta si es menor "
                        f"que {a}. Ojo: en la entropía, un valor más alto es MENOS concentración.")
        st.caption(f"Criterio para {nombre_corto}: {criterio} {fuente}")

        # Posición del caso real en la simulación Monte Carlo
        st.write(f"En la simulación, el caso real es **más concentrado que el "
                 f"{mas_concentrado_que:.1f} %** de los {n_sim:,} mercados aleatorios "
                 f"con {n} empresas.")

        # Lectura según el percentil, dividiendo en tercios
        if mas_concentrado_que < 100 / 3:
            nivel_percentil = "Baja"
            lectura = "está entre el tercio MENOS concentrado de los mercados simulados"
        elif mas_concentrado_que <= 200 / 3:
            nivel_percentil = "Moderada"
            lectura = "está en el tercio intermedio de los mercados simulados"
        else:
            nivel_percentil = "Alta"
            lectura = "está entre el tercio MÁS concentrado de los mercados simulados"

        # Si el umbral y el percentil no coinciden, se aclara por qué
        if nivel_percentil != correcta:
            st.info(
                f"Ojo: según el umbral la concentración es **{correcta.lower()}**, pero "
                f"comparado con la simulación el caso {lectura}. No es una contradicción: "
                f"el umbral mide la concentración en términos absolutos, mientras que el "
                f"percentil la compara con mercados al azar con el mismo número de "
                f"empresas ({n}). Con pocas empresas casi cualquier reparto es concentrado "
                f"en términos absolutos, aunque este mercado no lo sea tanto en relación "
                f"con lo esperable por azar (y al revés con muchas empresas)."
            )

        # Avisar si con este número de empresas ni el reparto más igualitario es "baja"
        iguales = np.full(n, 100 / n)
        nivel_minimo = clasificar(indicador, float(calcular_indicador(indicador, iguales, k)), n, k)
        if nivel_minimo != "Baja":
            st.caption(f"Nota: con {n} empresas, incluso si todas tuvieran la misma cuota la "
                       f"concentración sería {nivel_minimo.lower()} según este criterio.")
