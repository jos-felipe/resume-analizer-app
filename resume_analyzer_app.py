import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import tempfile
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração da página Streamlit
st.set_page_config(page_title="Analisador de Currículos", layout="wide")
st.title("Analisador de Currículos com IA")

# Inicialização das variáveis de sessão
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'groq_client' not in st.session_state:
    st.session_state.groq_client = None
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False

# Função para configurar o cliente Groq
def setup_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            st.session_state.groq_client = Groq(api_key=api_key)
            st.session_state.api_key_configured = True
            return True
        except Exception as e:
            st.error(f"Erro ao configurar o cliente Groq: {str(e)}")
    return False

# Tenta configurar o cliente Groq com a chave do .env
if not st.session_state.api_key_configured:
    setup_groq_client()

# Configuração dos embeddings
@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings()

embeddings = initialize_embeddings()

# Função para processar PDFs
def process_pdfs(pdf_files):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    all_chunks = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_path = temp_file.name
        
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        chunks = text_splitter.split_documents(pages)
        all_chunks.extend(chunks)
        os.unlink(temp_path)
    
    return all_chunks

# Função para criar ou atualizar o vector store
def update_vector_store(chunks):
    if st.session_state.vector_store is None:
        st.session_state.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
    else:
        st.session_state.vector_store.add_documents(chunks)

# Função para gerar resposta usando Groq
def generate_response(question):
    if st.session_state.vector_store is None:
        return "Por favor, faça upload de alguns currículos primeiro."
    
    if not st.session_state.api_key_configured:
        return "API Key do Groq não configurada."
    
    # Recupera os documentos relevantes
    docs = st.session_state.vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Prepara o prompt
    prompt = f"""Com base no seguinte contexto sobre currículos de candidatos, responda à pergunta.

Contexto:
{context}

Pergunta: {question}

Resposta:"""
    
    try:
        # Faz a chamada para a API do Groq
        completion = st.session_state.groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "Você é um assistente especializado em análise de currículos. Forneça respostas precisas e relevantes com base no contexto fornecido."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Erro ao gerar resposta: {str(e)}"

# Interface principal
if st.session_state.api_key_configured:
    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload de Currículos")
        uploaded_files = st.file_uploader(
            "Escolha os arquivos PDF",
            type="pdf",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Processar Currículos"):
                with st.spinner("Processando currículos..."):
                    chunks = process_pdfs(uploaded_files)
                    update_vector_store(chunks)
                    st.success(f"{len(uploaded_files)} currículo(s) processado(s) com sucesso!")

    with col2:
        st.header("Perguntas sobre os Candidatos")
        user_question = st.text_input("Faça uma pergunta sobre os candidatos:")
        
        if user_question:
            if st.button("Gerar Resposta"):
                with st.spinner("Gerando resposta..."):
                    response = generate_response(user_question)
                    st.write("Resposta:", response)

    # Área de informações adicionais
    st.markdown("---")
    st.markdown("""
    ### Como usar:
    1. Faça upload de um ou mais currículos em formato PDF
    2. Clique em "Processar Currículos"
    3. Faça perguntas sobre os candidatos na caixa de texto
    4. Clique em "Gerar Resposta" para obter análises baseadas nos currículos

    **Exemplos de perguntas:**
    - Quais candidatos têm experiência com Python?
    - Qual candidato tem mais experiência em gestão de equipes?
    - Liste os candidatos com formação em Ciência da Computação.
    """)
else:
    st.error("API Key do Groq não encontrada. Verifique se o arquivo .env está configurado corretamente.")
