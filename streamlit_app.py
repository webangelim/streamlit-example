import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os

# Configuração do Streamlit
st.set_page_config(page_title="InterrogaPPC-Inator", page_icon="./platypus.ico")

# Obtendo chave da OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY", st.secrets["ai"])

def setup_rag_chain():
    # Carrega o arquivo de texto
    loader = TextLoader("ppc.txt", autodetect_encoding=True)
    documents = loader.load()

    # Divide em pedaços menores
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500)
    chunks = text_splitter.split_documents(documents)

    # Cria os embeddings e armazena no FAISS (substituindo o Weaviate)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    retriever = vectorstore.as_retriever()

    # Template de prompt
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

    # Modelo de IA
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

    # Cadeia RAG
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def generate_response(input_text, rag_chain):
    return rag_chain.invoke(input_text)

st.title(":rainbow[InterrogaPPC-Inator]")

# Formulário de input
with st.form("my_form"):
    text = st.text_area("Digite sua pergunta:", "Quais as matérias recomendadas para os regulares de 2022 pegarem de acordo com o PPC?")
    submitted = st.form_submit_button("Enviar")

    if submitted:
        rag_chain = setup_rag_chain()
        st.info(generate_response(text, rag_chain))
