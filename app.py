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
chroma_client = chromadb.Client()
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
collection = chroma_client.create_collection(
    name="resumes",
    embedding_function=embedding_function
)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_resumes(zip_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        for filename in os.listdir(temp_dir):
            if filename.endswith('.pdf'):
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, 'rb') as pdf_file:
                    text = extract_text_from_pdf(pdf_file)
                    collection.add(
                        documents=[text],
                        metadatas=[{"filename": filename}],
                        ids=[filename]
                    )

def semantic_search(query, k=3):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    return results

def generate_answer(question, context):
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

def main():
    st.title("Resume Analyzer")
    
    # File uploader for ZIP file containing resumes
    uploaded_file = st.file_uploader("Upload ZIP file containing resumes (PDF format)", type="zip")
    
    if uploaded_file:
        with st.spinner("Processing resumes..."):
            process_resumes(uploaded_file)
        st.success("Resumes processed successfully!")
    
    # Question input
    question = st.text_input("Ask a question about the candidates:")
    
    if st.button("Generate Answer") and question:
        with st.spinner("Generating answer..."):
            search_results = semantic_search(question)
            context = "\n\n".join(search_results['documents'][0])
            answer = generate_answer(question, context)
            
            st.subheader("Answer:")
            st.write(answer)
            
            st.subheader("Sources:")
            for doc, metadata in zip(search_results['documents'][0], search_results['metadatas'][0]):
                st.write(f"- {metadata['filename']}")

if __name__ == "__main__":
    main()
