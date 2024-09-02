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
# from pinecone import Pinecone, ServerlessSpec


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

model_name = "gpt-4o"

client = OpenAI(api_key=OPENAI_API_KEY,)

model = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": "Dile al usuario que haga preguntas sobre el máster BIM de IDESIE",
        }
    ],
)
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

# pc = Pinecone(api_key=PINECONE_API_KEY)
# 
# pc.create_index(
#     name="quickstart",
#     dimension=2, # Replace with your model dimensions
#     metric="cosine", # Replace with your model metric
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ) 
# )





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
    ("Fundamentos BIM", "REVIT ARQ", "REVIT MEP", "Management Skills"),
)

# Status
#with st.status("Loading video..."):
#    st.write("Searching for data...")
#    time.sleep(2)
#    st.write("Found URL.")
#    time.sleep(1)
#    st.write("Downloading data...")
#    time.sleep(1)





################
# Pruebas sacadas de https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps#build-a-bot-that-mirrors-your-input

# Ventana de texto
with st.chat_message("assistant"):
    st.write("¡Hola!")

#prompt = st.chat_input("Escribe tu pregunta sobre el máster BIM de IDESIE")

# Inicializar el historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar los mensajes de chat del histórico al reacargar la aplicación
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Reaccionar a los mensajes del usuario
if prompt := st.chat_input(""):
    # Mostrar el mensaje del usuario en el contenedor del chat
    with st.chat_message("user"):
        st.markdown(prompt)
    # Añadir el mensaje al histórico de chat
    st.session_state.messages.append({"role": "user", "content": prompt})


###############





st.video(f"{YOUTUBE_VIDEO_URL}")
st.write(YOUTUBE_VIDEO_URL)

with st.expander("Transcripción (primeros 1000 caracteres)"):
     st.write(transcription_y[:1000])

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = OPENAI_API_KEY
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("Pregúntame lo que quieras"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
