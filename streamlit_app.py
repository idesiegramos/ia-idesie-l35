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

LOGO_URL_LARGE = "/img/logo.png"



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










