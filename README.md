# Simulador interactivo y evaluador de indicadores de concentración de mercado

## Qué hace la web

Aplicación web hecha con Streamlit que permite:

- Definir un mercado (caso real) con N empresas, escribiendo sus cuotas a mano o generándolas al azar con un botón.
- Calcular cuatro indicadores de concentración: razón de concentración `CR_k` (con k elegible), índice de Herfindahl-Hirschman (HHI), índice de dominancia (ID) e índice de entropía.
- Simular mercados aleatorios con el método de Monte Carlo (cuotas que suman 100 % en cada iteración) y graficar la distribución del indicador elegido.
- Marcar el caso real con una línea vertical sobre el histograma e indicar su percentil frente a la simulación.
- Evaluar al usuario: se le pregunta si la concentración del caso es baja, moderada o alta, y la página corrige la respuesta con umbrales y una justificación cuantitativa (valor, umbral aplicado y percentil).

## Generación de cuotas aleatorias (Monte Carlo)

- Se usa la distribución de Dirichlet con alpha = 1 para todas las empresas. Con alpha = 1, todos los repartos posibles de N cuotas son igual de probables (no se supone nada sobre la estructura del mercado) y cada mercado simulado suma exactamente 100 %.
- La simulación usa una semilla fija (42) para que los resultados sean reproducibles: los mismos datos dan siempre el mismo gráfico y percentil. El valor 42 es arbitrario. Con otra semilla los números cambian ligeramente, y ese efecto se reduce al aumentar las simulaciones.
- El botón "Generar caso real al azar" **no** usa semilla fija: cada clic genera un mercado distinto. El caso generado se conserva al cambiar de indicador.
