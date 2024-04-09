import os
# openai_api_key = os.getenv("OPENAI_API_KEY", "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")

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# ------ Imports da LLM ------ #
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

loader = TextLoader("ppc.txt", autodetect_encoding = True)
documents = loader.load()

# Divide em chunks de 500 palavras
text_splitter = CharacterTextSplitter("\n",chunk_size=500)
chunks = text_splitter.split_documents(documents)

client = weaviate.Client(
  embedded_options = EmbeddedOptions()
)

vectorstore = Weaviate.from_documents(
    client = client,
    documents = chunks,
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key),  #text-embedding-3-small usado para criar os embeddings
    by_text = False
)

retriever = vectorstore.as_retriever()

from langchain.prompts import ChatPromptTemplate

template = """Você é um assistente que irá responder perguntas.
Use as seguintes peças de texto para responder a pergunta.
Se não souber a resposta, apenas responda que não sabe a resposta.
Explique de forma concisa porém fornecendo um número considerável de detalhes.
Responda como se fosse um professor explicando para um aluno.
Pergunta: {question}
Contexto: {context}
Resposta:
"""
prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -----

# ----- Import do Streamlit ----- #
import streamlit as st
from langchain_openai import ChatOpenAI


st.title(':rainbow[InterrogaPPC-Inator]')

def generate_response(input_text):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    st.info(rag_chain.invoke(input_text))

with st.form('my_form'):
    text = st.text_area('Digite sua pergunta:', 'Como funcionam as horas de extensão?')
    submitted = st.form_submit_button('Enviar')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)

from langchain.prompts import ChatPromptTemplate

template = """Você é um assistente que irá responder perguntas.
Use as seguintes peças de texto para responder a pergunta.
Se não souber a resposta, apenas responda que não sabe a resposta.
Explique de forma concisa porém fornecendo um número considerável de detalhes.
Responda como se fosse um professor explicando para um aluno.
Pergunta: {question}
Contexto: {context}
Resposta:
"""
prompt = ChatPromptTemplate.from_template(template)
