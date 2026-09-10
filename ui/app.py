"""UI de chat en Streamlit para el RAG sobre Mondial Europe."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="RAG Mondial Europe", page_icon="globe_with_meridians")
st.title("RAG Mondial Europe")
st.caption("Asistente geográfico de Europa sobre un grafo de conocimiento RDF (SPARQL).")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Preguntá sobre países, capitales, ríos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando el grafo..."):
            try:
                resp = requests.post(f"{API_URL}/chat", json={"message": prompt}, timeout=180)
                resp.raise_for_status()
                reply = resp.json()["answer"]
            except Exception as exc:
                reply = f"Error al contactar la API: {exc}"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
