import streamlit as st
# import pandas as pd
# import math
# from pathlib import Path
# import tiktoken
import openai
import pytube
# import re


################################
# Secretos
################################

OPENAI_API_KEY = st.secrets.api_openai
PINECONE_API_KEY = st.secrets.api_pinecone
print("'Secretos' cargados correctamente")


################################
# Vídeo de prueba
################################

YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=dgZaIk3iFhc" # Clase MEP de IDESIE
# YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=ROax8vdhuEQ"

LOGO_URL_LARGE = "img/logo.png"



################################
# Vector store
################################









###############################################################################
# Construimos la página web
###############################################################################


# Logo
st.logo(
    LOGO_URL_LARGE,
    link="https://idesie.com"
)

# Pestañas
pages = {
    "Inteligencia IDESIE": [
        st.Page("inicio.py", title="Inicio"),
        st.Page("ia.py", title="Accede a nuestra IA"),
        st.Page("repositorio.py", title="Repositorio de clases"),
        st.Page("acerca_de.py", title="Acerca de"),
    ],
}

pg = st.navigation(pages)
pg.run()










