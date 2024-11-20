import streamlit as st
import time
import re
import os
import youtube_transcript_api
import getpass

from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA



################################
# Constantes (son variables que no cambian)
################################

YOUTUBE_VIDEO_URL : str = "https://www.youtube.com/watch?v=dgZaIk3iFhc"       # Clase MEP de IDESIE
# YOUTUBE_VIDEO_URL : str = "https://www.youtube.com/watch?v=ROax8vdhuEQ"



################################
# Secretos
################################

OPENAI_API_KEY = st.secrets["api_openai"]
PINECONE_API_KEY = st.secrets["api_pinecone"]

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

print("'Secretos' cargados correctamente")



################################
# Variables
################################

# Text Splitter
chunk_size: int = 8000
chunk_overlap: int = 200

# LLM (OpenAI)
model_name: str = "gpt-4o-mini"

# Embeddings (OpenAI Embeddings)
embedding_model: str = "text-embedding-3-small"

# Vector store (Pinecone)
index_name: str = "pinatest"
namespace: str = "idesie"
vector_store_dimension: int = 1536



################################
# Transcripción con youtube-transcript-api
################################

def extract_video_id(url):
    # Intenta extraer el ID del video de la URL estándar y corta
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&]+)",  # URL estándar
        r"(?:https?://)?youtu\.be/([^?]+)"                         # URL corta
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
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["es"])

        # Concatenar las partes de la transcripción en un solo texto
        transcript_text = "\n".join([entry['text'] for entry in transcript])
        return transcript_text

    except Exception as e:
        return str(e)

video_id = YOUTUBE_VIDEO_ID
content: str = get_transcript(video_id)

# with open("./transcripts/transcription_y.txt", "w", encoding="utf-8") as file:
#             file.write(transcription_y)



################################
# Text splitter
################################

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
)

content_splitted: list = text_splitter.create_documents([content])
print(f'Se ha dividido el contenido en {len(content_splitted)} partes (chunks) en la variable content_splitted, que es de tipo {type(content_splitted)}.\nCada elemento de la lista es de tipo {type(content_splitted[0])}.')







##############################################################################
# * * ########################################################################
##############################################################################
##############################################################################
##############################################################################

################################
# Modelo
################################

#model_name : str = "gpt-4o-mini"


# Configuración de la API Key de OpenAI
#client = OpenAI(api_key=OPENAI_API_KEY)


# Modelo por defecto
#if "openai_model" not in st.session_state:
#    st.session_state["openai_model"] = model_name


#model = client.chat.completions.create(
#    model=model_name,
#    messages=[
#        {
#            "role": "user",
#            "content": "Dile al usuario que haga preguntas sobre el máster BIM de IDESIE",
#        }
#    ],
#)

#print(f"Modelo '{model_name}' cargado correctamente")





################################################
# Langchain embeddings - OpenAI Embeddings
################################################

# client = OpenAI(api_key=OPENAI_API_KEY)
# 
# # Esto falla porque la transcripción supera el token máximo
# # embedding = client.embeddings.create(input=encoded_transcription, model="text-embedding-3-small")
# 
# #max_length = 100257
# max_length = 10
# 
# # Dividir el texto en partes más pequeñas si es necesario
# def dividir_texto(texto, max_length):
#     return [texto[i:i + max_length] for i in range(0, len(texto), max_length)]
# 
# # Crear embeddings para cada parte
# partes = dividir_texto(encoded_transcription, max_length)
# embeddings = []
# 
# for parte in partes:
#     embedding = client.embeddings.create(input=parte, model="text-embedding-3-small")
#     embeddings.append(embedding)





######################
# Pinecone
######################

#pc = Pinecone(api_key=PINECONE_API_KEY)


# Crear el índice
#index_name = "idesieindex"


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

# Crear el índice si no existe
#if index_name not in pc.list_indexes():
#    pc.create_index(
#        name=index_name,
#        dimension=1536,          # 1536 es la dimensión para el modelo 'text-embedding-3-small'
#        metric="cosine",
#        spec=ServerlessSpec(
#            cloud='aws',
#            region='us-east-1'
#        )
#    )
#
#
## Conectar al índice
#while not pc.describe_index(index_name).status['ready']:
#    time.sleep(1)
# 
#index = pc.Index(index_name)



######################
# Langchain
######################

# from langchain_core.documents import Document
# 
# documents = [
#     Document(
#         page_content="Dogs are great companions, known for their loyalty and friendliness.",
#         metadata={"source": "mammal-pets-doc"},
#     ),
#     Document(
#         page_content="Cats are independent pets that often enjoy their own space.",
#         metadata={"source": "mammal-pets-doc"},
#     ),
#     Document(
#         page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
#         metadata={"source": "fish-pets-doc"},
#     ),
#     Document(
#         page_content="Parrots are intelligent birds capable of mimicking human speech.",
#         metadata={"source": "bird-pets-doc"},
#     ),
#     Document(
#         page_content="Rabbits are social animals that need plenty of space to hop around.",
#         metadata={"source": "mammal-pets-doc"},
#     ),
# ]

#vectorstore = Chroma.from_documents(
#    documents,
#    embedding=OpenAIEmbeddings(),
#)





######################
# OpenAI embeddings
######################

#def vectorize_text(text):
#    """Vectoriza el texto usando el modelo de OpenAI."""
#    response = openai.Embedding.create(
#        input=[text],
#        model="text-embedding-3-small"
#    )
#    return response['data'][0]['embedding']
#
#def vectorize_and_store(text, metadata=None):
#    """Vectoriza el texto y lo guarda en Pinecone."""
#    # Vectorizar el texto
#    vector = vectorize_text(text)
#    
#    # Generar un ID único (puedes adaptar esto según tus necesidades)
#    id = f"doc_{len(index.fetch([])['vectors'])}"
#    
#    # Guardar en Pinecone
#    index.upsert(vectors=[(id, vector, metadata)])
#    
#    return id
#
#
## Ejemplo de uso
#transcript = "Este es un ejemplo de transcripción de un video de YouTube."
#metadata = {"source": "YouTube", "video_id": "Revit_MEP_2023-10-27"}

#doc_id = vectorize_and_store(transcript, metadata)
#print(f"Documento guardado con ID: {doc_id}")


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

#st.write(vectorstore.similarity_search("cat"))

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
     st.write(content[:1000])

# with st.expander("encoded_transcription"):
#      st.write(encoded_transcription[:1000])

with st.expander("embeddings"):
     st.write(embeddings[:1000])

