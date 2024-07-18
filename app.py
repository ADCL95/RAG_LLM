import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import chromadb
from langchain.vectorstores import Chroma
from chromadb.utils import embedding_functions
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from chromadb.config import Settings

OPENAI_API_KEY = 'sk-proj-nm1bnhxaWh3NYT2nZAGWT3BlbkFJ1dyTpe5ZBTmkVaUM0Prd'

# Configurar la página de Streamlit
st.set_page_config(page_title='Comision de la verdad')

# Cargar imagen
image_path = "logo_comision.jpeg"  # Cambia esta ruta a la ubicación de tu imagen
col1, col2 = st.columns([1, 5])
with col1:
    st.image(image_path, use_column_width=True)
with col2:
    st.header("CHAT BOT COMISION DE LA VERDAD")

# Inicializar modelo
llm_name = "gpt-3.5-turbo-0301"
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=llm_name, temperature=0)

# Configurar embeddings y Chroma
embedding_function = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
chroma_client = chromadb.HttpClient(host='54.163.27.67', port=8000)

langchain_chroma = Chroma(
    client=chroma_client,
    collection_name="Comision_verdad",
    embedding_function=embedding_function
)

# Definir prompt
template = """
Utiliza los siguientes elementos de contexto para responder a la pregunta al final. Si no conoces la respuesta, intenta buscar en toda la informacion la similaridad en los textos. 
Usa un máximo de tres oraciones. Eres un asistente virtual que responde preguntas de la comision de la verdad en colombia este Entidad de Estado que busca el esclarecimiento de los patrones y
causas explicativas del conflicto armado interno que satisfaga el derecho de las víctimas y de la sociedad a la verdad, promueva el reconocimiento de lo sucedido, la convivencia en los territorios y contribuya a sentar las bases para la no repetición, mediante un proceso de participación amplio y plural para la construcción de una paz estable y duradera,
en ellos se escriben diversos contextos en relacion a los derechos humanos. En la respuesta que me des trata de identificar actores, anos y delitos que se cometieron a victimas. Siempre di "¡gracias por preguntar!" al final de la respuesta.
{context}
Question: {question}
Respuesta útil:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Cachear la función para ejecutar la cadena QA
@st.cache
def run_qa_chain(user_question):
    qa_chain_refine = RetrievalQA.from_chain_type(
        llm,
        retriever=langchain_chroma.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result = qa_chain_refine({"query": user_question})
    return result["result"]

# Crear entrada de texto para la pregunta del usuario
user_question = st.text_input("Pregunta")

# Mostrar el resultado cuando se presiona el botón
if st.button("Enviar pregunta"):
    result = run_qa_chain(user_question)
    st.write(result)
