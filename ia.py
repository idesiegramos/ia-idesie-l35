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
comprobacion_splitted = 'Se ha dividido el contenido en ' + str(len(content_splitted)) + ' partes (chunks) en la variable content_splitted, que es de tipo ' + str(type(content_splitted)) + '.\nCada elemento de la lista es de tipo ' + str(type(content_splitted[0])) + '.'




###############################
# Pinecone
###############################

# Creamos el cliente de Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Lista de índices existentes en Pinecone
existing_indexes: list[str] = pc.list_indexes().names()
print(f'En Pinecone existen los siguientes índices: {existing_indexes}')

# Creamos el índice solo si no existe ya
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=vector_store_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    print(f'Índice "{index_name}" creado.')
else:
    print(f'El índice "{index_name}" ya existe, por lo que no se creará de nuevo.')


# Creamos el indexador de LangChain (el generador de embeddings de OpenAI), para transformar texto a su representación vectorial:
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=embedding_model)


# Generamos los embeddings y los subimos al vector store de Pinecone
# Ojo, se suben cada vez que se ejecuta el código
# Habría que subirlos solo si no están ya en el vector store
vectorstore_from_docs = PineconeVectorStore.from_documents(
        content_splitted,
        index_name=index_name,
        namespace=namespace,
        embedding=embeddings
    )


################################
# LLM (OpenAI)
################################

# Preparamos el modelo LLM para preguntas y respuestas
llm = ChatOpenAI(
    model_name=model_name,
    temperature=0.3  # Ajusta la creatividad de la respuesta
)

# Conectar al índice de Pinecone
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace
)

# Crear el retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",  # Puedes cambiar a "mmr" para diversidad
    search_kwargs={"k": 4}  # Número de documentos a recuperar
)

# Crear cadena de recuperación de QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Método de inserción de contexto
    retriever=retriever,
    return_source_documents=False  # Opcional: devuelve documentos fuente
)

# Función para hacer preguntas
def hacer_pregunta(pregunta):
    # Realizar la pregunta
    resultado = qa_chain({"query": pregunta})

    # Imprimir respuesta
    return f"Respuesta: {resultado["result"]}"

    # Imprimir documentos fuente (si está habilitado)
    if "source_documents" in resultado:
        print("\nDocumentos fuente:")
        for doc in resultado["source_documents"]:
            print(f"- {doc.page_content[:200]}...")  # Muestra extracto del documento


# Ejemplo de uso
pregunta = "¿Qué tipos de conductos se mencionan?"
respuesta_llm_test = hacer_pregunta(pregunta)




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
if prompt := st.chat_input("Escribe tu pregunta, majo"):
    
    # Añadir el mensaje al histórico de chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Mostrar el mensaje del usuario en el contenedor del chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # # Mostrar la respuesta del asistente en el contenedor de chat
    # with st.chat_message("assistant"):
    #     stream = client.chat.completions.create(
    #         model=st.session_state["openai_model"],
    #         messages=[
    #             {"role": m["role"], "content": m["content"]}
    #             for m in st.session_state.messages
    #         ],
    #         stream=True,
    #     )
    #     response = st.write_stream(stream)
    # st.session_state.messages.append({"role": "assistant", "content": response})

    # Mostrar la respuesta del asistente en el contenedor de chat
    with st.chat_message("assistant"):
        # Contenedor para mostrar la respuesta
        message_placeholder = st.empty()

        # Llamar a la cadena de QA
        try:
            resultado = qa_chain({"query": prompt})
            respuesta = resultado['result']

            # Mostrar respuesta con efecto de escritura
            full_response = ""
            for chunk in respuesta.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
                
            message_placeholder.markdown(full_response)

            # Añadir respuesta al historial
            st.session_state.messages.append({
                "role": "assistant", 
                "content": respuesta
            })

            # Mostrar documentos fuente si están disponibles
            if resultado.get('source_documents'):
                with st.expander("Documentos fuente"):
                    for doc in resultado['source_documents']:
                        st.write(doc.page_content[:300] + "...")

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")


#
#
###############################





st.video(f"{YOUTUBE_VIDEO_URL}")
st.write(YOUTUBE_VIDEO_URL)

with st.expander("Transcripción (primeros 1000 caracteres)"):
     st.write(content[:1000])

# with st.expander("encoded_transcription"):
#      st.write(encoded_transcription[:1000])

# with st.expander("embeddings"):
#      st.write(embeddings[:1000])

st.write(comprobacion_splitted)

st.write(respuesta_llm_test)