import streamlit as st
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente
load_dotenv()

# Configurar o Streamlit
st.set_page_config(page_title="Analisador de Currículos", page_icon="📄")
st.title("Analisador de Currículos com RAG")

# Função para extrair texto de PDF
def extract_text_from_pdf(file):
    pdf_reader = pypdf.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Função para processar o currículo
def process_resume(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    return vector_store

# Configurar o modelo de linguagem
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Interface do Streamlit
uploaded_file = st.file_uploader("Faça upload do currículo (PDF)", type="pdf")

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    vector_store = process_resume(resume_text)
    
    st.success("Currículo processado com sucesso!")
    
    # Criar a cadeia de RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    
    # Interface para perguntas e respostas
    question = st.text_input("Faça uma pergunta sobre o currículo:")
    if question:
        response = qa_chain({"query": question})
        st.write("Resposta:", response["result"])
        
        st.write("Fontes:")
        for doc in response["source_documents"]:
            st.write(doc.page_content)

st.sidebar.title("Sobre")
st.sidebar.info(
    "Esta aplicação usa Streamlit, FAISS e um modelo de linguagem para analisar "
    "currículos e responder perguntas sobre as qualificações dos candidatos."
)
