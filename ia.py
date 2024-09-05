import streamlit as st
import time
import re
import openai
import tiktoken
import youtube_transcript_api

from openai import OpenAI
#from dotenv import load_dotenv
from tiktoken import encoding_for_model
#from langchain_openai.chat_models import ChatOpenAI
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
#from langchain_openai.embeddings import OpenAIEmbeddings
#from langchain_community.vectorstores import DocArrayInMemorySearch
#from langchain_core.runnables import RunnableParallel, RunnablePassthrough
#from pinecone.grpc import PineconeGRPC as Pinecone
#from pinecone import ServerlessSpec




################################
# Constantes (son variables que no cambian)
################################

YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=dgZaIk3iFhc" # Clase MEP de IDESIE
# YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=ROax8vdhuEQ"



################################
# Secretos
################################

OPENAI_API_KEY = st.secrets.api_openai
PINECONE_API_KEY = st.secrets.api_pinecone
print("'Secretos' cargados correctamente")



################################
# Modelo
################################

model_name = "gpt-4o-mini-2024-07-18"


# Configuración de la API Key de OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Modelo por defecto
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = model_name


#model = client.chat.completions.create(
#    model=model_name,
#    messages=[
#        {
#            "role": "user",
#            "content": "Dile al usuario que haga preguntas sobre el máster BIM de IDESIE",
#        }
#    ],
#)

print(f"Modelo '{model_name}' cargado correctamente")




################################
# Transcripción con youtube-transcript-api
################################

def extract_video_id(url):
    # Intenta extraer el ID del video de la URL estándar y corta
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&]+)',  # URL estándar
        r'(?:https?://)?youtu\.be/([^?]+)'                         # URL corta
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("URL de video no válida")

# Reemplaza 'your_video_url' con la URL del video de YouTube
video_url = YOUTUBE_VIDEO_URL
YOUTUBE_VIDEO_ID = extract_video_id(video_url)
print(f"ID del vídeo de YouTube: {YOUTUBE_VIDEO_ID}")


def get_transcript(video_id):
    try:
        # Obtener la transcripción del video
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['es'])

        # Concatenar las partes de la transcripción en un solo texto
        transcript_text = "\n".join([entry['text'] for entry in transcript])
        return transcript_text

    except Exception as e:
        return str(e)

video_id = YOUTUBE_VIDEO_ID
transcription_y = get_transcript(video_id)
with open("./transcripts/transcription_y.txt", "w", encoding="utf-8") as file:
            file.write(transcription_y)


######################
# Pinecone
######################

#pc = Pinecone(api_key=PINECONE_API_KEY)

# Crear el índice
#index_name = "idesieindex"
#
#if not pc.Index(index_name):
#    pc.create_index(
#        name=index_name,
#        dimension=1536,
#        metric="cosine",
#        spec=ServerlessSpec(
#            cloud='aws',
#            region='us-east-1'
#        )
#    )
#
#index = pc.Index(index_name)




########################################################################
########################################################################
# A partir de aquí se diseña la página web
########################################################################
########################################################################

# Título y descripción

st.title("IA de IDESIE Business&Tech School")
st.write(
    "Esta Inteligencia Artificial te ayudará a con el contenido de las clases impartidas en IDESIE. "
    "Para usar esta aplicación, solo debes escribir en el chat la información que quieres obtener, como por ejemplo un resumen con los puntos importantes de la clase o resolver otro tipo de dudas relacionadas con el contenido del vídeo. "
    "Te será de ayuda."
)


option = st.selectbox(
    "Asignatura",
    ("Todas las asignaturas", 
    "Bases de Datos", 
    "BIM & Facility Management",
    "BIM Construction",
    "BIM Design - L35",
    "BIM Project Management. Costes",
    "BIM Project Management. Plan de Gerencia",
    "BIM Project Management. Planificación",
    "Civil 3D e Infraworks",
    "Dynamo",
    "Estrategia",
    "Instalaciones con Trox Technik",
    "Management Skills",
    "Mediciones y Presupuestos",
    "Navisworks",
    "Plan de Gerencia",
    "Protocolos y Estándares BIM",
    "Revit Arquitectura", 
    "Revit MEP", 
    "Trabajo colaborativo. CDE"),
)
# Status
#with st.status("Loading video..."):
#    st.write("Searching for data...")
#    time.sleep(2)
#    st.write("Found URL.")
#    time.sleep(1)
#    st.write("Downloading data...")
#    time.sleep(1)





################################
# Pruebas sacadas de https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps#build-a-bot-that-mirrors-your-input
#

# Ventana de texto
#with st.chat_message("assistant"):
#    st.write("¡Hola!")

#prompt = st.chat_input("Escribe tu pregunta sobre el máster BIM de IDESIE")


# Inicializar el historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar los mensajes de chat del histórico al recargar la aplicación
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Reaccionar a los mensajes del usuario
if prompt := st.chat_input("Escribe tu pregunta"):
    
    # Añadir el mensaje al histórico de chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Mostrar el mensaje del usuario en el contenedor del chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Mostrar la respuesta del asistente en el contenedor de chat
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})


#
#
###############################





st.video(f"{YOUTUBE_VIDEO_URL}")
st.write(YOUTUBE_VIDEO_URL)

with st.expander("Transcripción (primeros 1000 caracteres)"):
     st.write(transcription_y[:1000])

