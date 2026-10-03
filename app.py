import streamlit as st

st.title("Simulador de concentración de mercado")

n = st.number_input("Número de empresas", min_value=2, max_value=100, value=4)
indicador = st.selectbox("Indicador", ["CR", "IHH", "ID", "IE"])

st.write(f"Elegiste {indicador} con {n} empresas")
