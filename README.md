# Resume Analizer with RAG

This project implements a resume analyzer using Streamlit, natural language processing, and Retrieval-Augmented Generation (RAG). The application allows users to upload resumes in PDF format, process them, and ask questions about the candidates' qualifications.

## Main Components

1. **Streamlit**: Framework for creating the web interface of the application.
2. **PyPDF**: Library for extracting text from PDF files.
3. **LangChain**: Set of tools for natural language processing and RAG.
4. **FAISS**: Library for efficient search of similar vectors.
5. **Hugging Face**: Platform for accessing pre-trained language models.

## Workflow

1. **Resume Upload**: The user uploads a PDF file containing the resume.

2. **Text Extraction**: The text is extracted from the PDF using the PyPDF library.

3. **Text Processing**:
	- The text is split into smaller chunks using `RecursiveCharacterTextSplitter`.
	- Each chunk is converted into an embedding using `HuggingFaceEmbeddings`.
	- The embeddings are stored in a FAISS index for efficient search.

4. **Language Model Configuration**: The "google/flan-t5-large" model from Hugging Face is used to generate responses.

5. **Question and Answer Interface**:
	- The user types a question about the resume.
	- The question is processed by the RAG system:
	  a. Searches for the most relevant chunks in the FAISS index.
	  b. Uses the language model to generate a response based on the retrieved chunks and the question.
	- The response is displayed to the user, along with the resume excerpts used as sources.

## Technologies Used

- **Python**: Main programming language.
- **Streamlit**: To create the interactive web interface.
- **LangChain**: To implement the RAG pipeline.
- **FAISS**: For efficient storage and search of embeddings.
- **Hugging Face**: To access pre-trained language models.
- **PyPDF**: To extract text from PDF files.

## Advantages of the RAG Approach

1. **Contextualized Responses**: Responses are generated based on the specific content of the resume.
2. **Flexibility**: Can answer a wide variety of questions without specific training.
3. **Transparency**: Shows the sources of the information used to generate the responses.
4. **Efficiency**: Uses vector search to quickly find relevant information in long resumes.

## Possible Improvements

1. **Support for Multiple Resumes**: Allow the upload and analysis of multiple resumes simultaneously.
2. **Comparative Analysis**: Implement features to compare different candidates.
3. **Structured Information Extraction**: Add the capability to extract and organize specific information (e.g., skills, experience) in a structured manner.
4. **Enhanced User Interface**: Add more advanced visualizations and filters to improve the user experience.
5. **More Advanced Language Models**: Experiment with newer and more powerful models to improve the quality of responses.

This project demonstrates a practical application of advanced NLP and AI techniques to solve a real problem of resume analysis, offering a powerful tool for HR professionals and recruiters.

