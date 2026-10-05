Link: https://claude.ai/share/40267ebd-2bbc-4755-95ee-b60424c6a6af

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

Requisitos:
Python 3.9 o superior. Puedes comprobar tu versión con python --version (en macOS/Linux puede ser python3 --version). Si no lo tienes, descárgalo desde python.org. En Windows, durante la instalación marca la casilla "Add Python to PATH".
Git (opcional), para clonar el repositorio. Si no lo tienes, puedes descargar el proyecto como ZIP (ver paso 1).
Ejecutar la app en tu computador
1. Descargar el proyecto

Con Git:

bash
git clone https://github.com/martin-san-juan/taller.git
cd taller

Sin Git: en la página del repositorio pulsa Code → Download ZIP, descomprime el archivo y abre una terminal dentro de la carpeta.

2. Crear un entorno virtual (recomendado)

Un entorno virtual instala las librerías solo para este proyecto, sin mezclarlas con otros.

En Windows:

bash
python -m venv venv
venv\Scripts\activate

En macOS / Linux:

bash
python3 -m venv venv
source venv/bin/activate

Sabrás que está activo porque la terminal mostrará (venv) al inicio de la línea. Cada vez que vuelvas a abrir una terminal para usar la app, tendrás que activarlo de nuevo con la segunda línea.

3. Instalar las dependencias
bash
pip install -r requirements.txt

Esto instala streamlit, numpy y matplotlib.

4. Ejecutar la app
bash
streamlit run app.py

Se abrirá automáticamente una pestaña en el navegador con la app. Si no se abre, entra a mano en http://localhost:8501.

Importante: usa streamlit run app.py, no python app.py. Con python el código se ejecuta pero no se abre ninguna página.

5. Detener la app

En la terminal pulsa Ctrl + C. Para salir del entorno virtual escribe deactivate.

Uso rápido
Indica el número de empresas (N) y escribe sus cuotas de mercado en %, separadas por comas (deben sumar 100), o pulsa 🎲 Generar caso real al azar.
En la barra lateral elige el indicador a analizar (y k si eliges CR_k) y el número de simulaciones.
Revisa el histograma, el percentil y la interpretación, y responde la pregunta del final.
