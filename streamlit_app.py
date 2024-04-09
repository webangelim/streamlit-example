import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os

st.set_page_config(page_title = "InterrogaPPC-Inator", page_icon="./platypus.ico")

openai_api_key = os.getenv("OPENAI_API_KEY", st.secrets["ai"])

@st.cache(allow_output_mutation=True)
def setup_rag_chain():
    loader = TextLoader("ppc.txt", autodetect_encoding=True)
    documents = loader.load()
    text_splitter = CharacterTextSplitter("\n", chunk_size=500)
    chunks = text_splitter.split_documents(documents)
    client = weaviate.Client(embedded_options=EmbeddedOptions())
    vectorstore = Weaviate.from_documents(
        client=client,
        documents=chunks,
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=openai_api_key
        ),  # text-embedding-3-small usado para criar os embeddings
        by_text=False,
    )
    retriever = vectorstore.as_retriever()
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

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key
    )
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def generate_response(input_text, rag_chain):
    return rag_chain.invoke(input_text)

# st.title(":orange[InterrogaPPC-Inator]")

# Dividir a tela em duas colunas
col1, col2 = st.columns([1, 1])

# Adicionar o título na primeira coluna
with col1:
    st.title(":orange[InterrogaPPC-Inator]")

# Adicionar a imagem do ornitorrinco na segunda coluna
with col2:
    st.image("platypus.png")

with st.form("my_form"):
    text = st.text_area("Digite sua pergunta:", "Como funcionam as horas de extensão?")
    submitted = st.form_submit_button("Enviar")
    if submitted:
        rag_chain = setup_rag_chain()
        st.info(generate_response(text, rag_chain))
