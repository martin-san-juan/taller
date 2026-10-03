"""
Simulación Monte Carlo de la concentración de un mercado (índice HHI)
y comparación con un caso particular.

Requisitos:  pip install numpy matplotlib
Ejecutar:    python simulacion_concentracion.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # dibuja sin abrir ventanas (útil en un servidor web)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# 1. PARÁMETROS
# ---------------------------------------------------------------
N_SIMULACIONES = 10_000   # cuántos mercados aleatorios generar
ALPHA = 1.0               # qué tan desiguales son los mercados simulados
SEMILLA = 42              # fija el azar para obtener siempre el mismo resultado

# Caso particular: cuota de mercado de cada empresa (en %, debe sumar 100)
CUOTAS_REALES = [40, 25, 15, 12, 8]


# ---------------------------------------------------------------
# 2. MEDIDAS DE CONCENTRACIÓN
# ---------------------------------------------------------------
def calcular_hhi(cuotas):
    """HHI = suma de los cuadrados de las cuotas (en %). Va de ~0 a 10.000."""
    cuotas = np.asarray(cuotas, dtype=float)
    return np.sum(cuotas ** 2, axis=-1)


def calcular_cr(cuotas, k=4):
    """CRk = suma de las cuotas de las k empresas más grandes."""
    cuotas = np.asarray(cuotas, dtype=float)
    ordenadas = -np.sort(-cuotas, axis=-1)      # ordena de mayor a menor
    return ordenadas[..., :k].sum(axis=-1)


def clasificar_hhi(hhi):
    """Umbrales de las Merger Guidelines de EE.UU. (2023)."""
    if hhi < 1000:
        return "No concentrado"
    elif hhi <= 1800:
        return "Moderadamente concentrado"
    return "Altamente concentrado"


# ---------------------------------------------------------------
# 3. SIMULACIÓN MONTE CARLO
# ---------------------------------------------------------------
def simular_mercados(n_empresas, n_sim, alpha=1.0, semilla=None):
    """Genera n_sim mercados aleatorios de n_empresas cuyas cuotas suman 100 %."""
    rng = np.random.default_rng(semilla)
    proporciones = rng.dirichlet(np.full(n_empresas, alpha), size=n_sim)
    return proporciones * 100                   # matriz (n_sim filas, n_empresas columnas)


# ---------------------------------------------------------------
# 4. COMPARACIÓN CON EL CASO REAL
# ---------------------------------------------------------------
def analizar_mercado(cuotas_reales, n_sim=N_SIMULACIONES, alpha=ALPHA, semilla=SEMILLA):
    cuotas_reales = np.asarray(cuotas_reales, dtype=float)
    if not np.isclose(cuotas_reales.sum(), 100):
        raise ValueError(f"Las cuotas deben sumar 100 (suman {cuotas_reales.sum():.2f}).")

    n = len(cuotas_reales)

    # Mercados simulados y sus índices
    simulados = simular_mercados(n, n_sim, alpha, semilla)
    hhi_sim = calcular_hhi(simulados)
    cr4_sim = calcular_cr(simulados, k=4)

    # Índices del caso real
    hhi_real = float(calcular_hhi(cuotas_reales))
    cr4_real = float(calcular_cr(cuotas_reales, k=4))

    # ¿Qué % de mercados simulados es MENOS o igual de concentrado que el real?
    percentil = float(np.mean(hhi_sim <= hhi_real) * 100)
    p025, p975 = np.percentile(hhi_sim, [2.5, 97.5])

    resultados = {
        "n_empresas": n,
        "hhi_real": round(hhi_real, 1),
        "clasificacion_real": clasificar_hhi(hhi_real),
        "cr4_real": round(cr4_real, 1),
        "hhi_simulado_media": round(float(hhi_sim.mean()), 1),
        "hhi_simulado_desv": round(float(hhi_sim.std()), 1),
        "hhi_intervalo_95": [round(float(p025), 1), round(float(p975), 1)],
        "cr4_simulado_media": round(float(cr4_sim.mean()), 1),
        "percentil_caso_real": round(percentil, 1),
    }
    return resultados, hhi_sim


# ---------------------------------------------------------------
# 5. GRÁFICO
# ---------------------------------------------------------------
def graficar(hhi_sim, hhi_real, archivo="hhi_montecarlo.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(hhi_sim, bins=60, color="steelblue", alpha=0.7, label="Mercados simulados")
    ax.axvline(hhi_real, color="crimson", linewidth=2,
               label=f"Caso real (HHI = {hhi_real:.0f})")
    ax.set_xlabel("Índice HHI")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución Monte Carlo del HHI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(archivo, dpi=120)
    plt.close(fig)
    return archivo


# ---------------------------------------------------------------
# 6. PROGRAMA PRINCIPAL
# ---------------------------------------------------------------
if __name__ == "__main__":
    resultados, hhi_sim = analizar_mercado(CUOTAS_REALES)

    for clave, valor in resultados.items():
        print(f"{clave:>22}: {valor}")

    print(f"\nEl caso real es más concentrado que el "
          f"{resultados['percentil_caso_real']}% de los mercados simulados.")

    archivo = graficar(hhi_sim, resultados["hhi_real"])
    print(f"Gráfico guardado en: {archivo}")
