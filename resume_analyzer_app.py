import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import tempfile
import os
from dotenv import load_dotenv
import spacy
import re
from typing import Optional

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

# Carrega o modelo em português. Se não estiver instalado, você precisa executar:
# python -m spacy download pt_core_news_lg
try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    print("Modelo spaCy não encontrado. Instalando...")
    import os
    os.system("python -m spacy download pt_core_news_lg")
    nlp = spacy.load("pt_core_news_lg")

def extract_candidate_name(doc_content: str) -> Optional[str]:
    # Lista de possíveis indicadores de nome em um currículo
    name_indicators = [
        r"nome:[\s\n]*([\w\s]+)",
        r"nome completo:[\s\n]*([\w\s]+)",
        r"curriculum vitae[\s\n]+(de)?[\s\n]*([\w\s]+)",
        r"currículo[\s\n]+(de)?[\s\n]*([\w\s]+)"
    ]
    
    # Primeiro, tenta encontrar o nome usando os indicadores
    for pattern in name_indicators:
        match = re.search(pattern, doc_content.lower())
        if match:
            groups = [g for g in match.groups() if g]
            potential_name = groups[-1].strip()
            if len(potential_name.split()) >= 2:
                return potential_name.title()
    
    # Se não encontrar usando indicadores, usa spaCy para identificar nomes próprios
    doc = nlp(doc_content[:1000])  # Limita a 1000 caracteres por eficiência
    
    # Lista para armazenar possíveis nomes encontrados
    potential_names = []
    
    # Procura por entidades do tipo PESSOA
    for ent in doc.ents:
        if ent.label_ == "PER":
            potential_names.append((ent.text, ent.start_char))
    
    if potential_names:
        # Ordena os nomes encontrados pela posição no texto (assume que o primeiro nome é mais provável de ser o do candidato)
        potential_names.sort(key=lambda x: x[1])
        # Retorna o primeiro nome encontrado que tenha pelo menos duas palavras
        for name, _ in potential_names:
            if len(name.split()) >= 2:
                return name
    
    return None

def generate_response(question):
    if st.session_state.vector_store is None:
        return "Por favor, faça upload de alguns currículos primeiro."
    
    if not st.session_state.api_key_configured:
        return "API Key do Groq não configurada."
    
    # Recupera os documentos relevantes
    docs = st.session_state.vector_store.similarity_search(question, k=3)
    
    # Prepara o contexto com identificação clara dos candidatos
    formatted_context = ""
    for i, doc in enumerate(docs, 1):
        # Tenta extrair o nome real do documento usando spaCy
        candidate_name = extract_candidate_name(doc.page_content)
        
        # Se não conseguir extrair um nome, usa um identificador numérico
        if not candidate_name:
            candidate_name = f"Candidato {i}"
        
        formatted_context += f"\n{candidate_name}:\n{doc.page_content}\n"
    
    # Prompt atualizado
    prompt = f"""Analise as informações dos seguintes candidatos e responda à pergunta, identificando claramente cada candidato em sua resposta.

Contexto dos Currículos:
{formatted_context}

Pergunta: {question}

Por favor, forneça uma resposta que:
1. Identifique especificamente cada candidato relevante
2. Cite as experiências ou habilidades específicas relacionadas à pergunta
3. Se não houver informações suficientes sobre algum candidato, mencione isso
4. Se aplicável, compare as experiências entre os candidatos

Resposta:"""
    
    try:
        completion = st.session_state.groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "Você é um especialista em análise de currículos. Forneça respostas detalhadas e precisas, sempre identificando claramente cada candidato e suas respectivas qualificações."
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
        
                    # Mostra um resumo dos currículos processados
                    with st.expander("Ver resumo dos currículos processados"):
                        for i, file in enumerate(uploaded_files, 1):
                            st.write(f"**Currículo {i}:** {file.name}")
                            # Você pode adicionar mais informações aqui se desejar

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
