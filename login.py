import streamlit as st

st.title("Login")
st.write("Inicia sesión con tu cuenta de IDESIE")

text_input = st.text_input(
    "Correo de IDESIE:",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
    placeholder="usuario@idesie.com",
)

    if text_input:
        st.write("Tu usuario es: ", text_input)
st.button("Iniciar sesión", type="primary")