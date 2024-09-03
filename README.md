# multiPDFbot_local

## Interactive Document Q&A
This project implements an interactive question-and-answer system that allows users to ask questions about documents using a conversational AI model. It utilizes Langchain and OpenAI's API to process documents and generate responses.

## Features
Load and process various document types (PDF, DOCX).
Conversational retrieval of information from documents.
Fallback responses for common questions.
User-friendly interface built with Streamlit.

## Installation
Clone the repository:

git clone cd

## Run the Streamlit app:

streamlit run app.py

Open your browser and go to http://localhost:8501 to interact with the app.

## How It Works
- The application loads documents from the folder and processes them using Langchain's document loaders.
- It splits the documents into manageable chunks and generates embeddings for retrieval.
- Users can ask questions, and the system retrieves relevant answers based on the processed documents.
## Instructions for Use
- Replace <repository-url> and <repository-directory> with your actual repository details.
- Ensure that the requirements.txt file includes all necessary dependencies.
- Adjust any section as needed based on your specific project requirements.
- This template provides a clear and comprehensive overview of the project, making it easier for others to understand and contribute.



