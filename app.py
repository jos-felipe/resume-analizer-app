import streamlit as st
import chromadb
import os
import tempfile
import zipfile
import PyPDF2
from groq import Groq
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize ChromaDB
@st.cache_resource
def init_chroma():
    chroma_client = chromadb.Client()
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
    
    # Get all collection names
    collection_names = [col.name for col in chroma_client.list_collections()]
    
    # If 'resumes' collection exists, get it; otherwise, create it
    if "resumes" in collection_names:
        collection = chroma_client.get_collection(
            name="resumes",
            embedding_function=embedding_function
        )
    else:
        collection = chroma_client.create_collection(
            name="resumes",
            embedding_function=embedding_function
        )
    
    return collection

# Get the collection
collection = init_chroma()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_resumes(zip_file):
    try:
        # Clear existing data
        collection.delete(where={})
    except Exception as e:
        st.error(f"Error clearing existing data: {str(e)}")
        return False
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            documents = []
            metadatas = []
            ids = []
            
            for filename in os.listdir(temp_dir):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(temp_dir, filename)
                    with open(file_path, 'rb') as pdf_file:
                        text = extract_text_from_pdf(pdf_file)
                        documents.append(text)
                        metadatas.append({"filename": filename})
                        ids.append(filename)
            
            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                return True
            else:
                st.warning("No PDF files found in the uploaded ZIP.")
                return False
    except Exception as e:
        st.error(f"Error processing resumes: {str(e)}")
        return False

def semantic_search(query, k=3):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=k
        )
        return results
    except Exception as e:
        st.error(f"Error during semantic search: {str(e)}")
        return None

def generate_answer(question, context):
    try:
        prompt = f"""You are a recruitment assistant analyzing resumes. Based on the following context from resumes, answer this question: {question}

Context:
{context}

Answer:"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return None

def main():
    st.title("Resume Analyzer")
    
    # File uploader for ZIP file containing resumes
    uploaded_file = st.file_uploader("Upload ZIP file containing resumes (PDF format)", type="zip")
    
    if uploaded_file:
        with st.spinner("Processing resumes..."):
            success = process_resumes(uploaded_file)
        if success:
            st.success("Resumes processed successfully!")
    
    # Question input
    question = st.text_input("Ask a question about the candidates:")
    
    if st.button("Generate Answer") and question:
        with st.spinner("Generating answer..."):
            search_results = semantic_search(question)
            
            if not search_results or not search_results['documents'][0]:
                st.warning("No relevant information found. Please ensure resumes are uploaded and try a different question.")
                return
                
            context = "\n\n".join(search_results['documents'][0])
            answer = generate_answer(question, context)
            
            if answer:
                st.subheader("Answer:")
                st.write(answer)
                
                st.subheader("Sources:")
                for doc, metadata in zip(search_results['documents'][0], search_results['metadatas'][0]):
                    st.write(f"- {metadata['filename']}")

if __name__ == "__main__":
    main()
