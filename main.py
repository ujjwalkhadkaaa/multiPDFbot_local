import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

def process_document_from_folder(folder_path, file_type):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(f".{file_type.lower()}"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "rb") as f:
                if file_type == "PDF":
                    loader = PyPDFLoader(file_path)
                elif file_type == "DOCX":
                    loader = UnstructuredWordDocumentLoader(file_path)
                else:  # Text file
                    loader = TextLoader(file_path)
                documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    
    return db

def get_conversation_chain(vector_store):
    llm = OpenAI()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
    )
    return conversation_chain

# Fallback response for basic questions
def get_fallback_response(question):
    basic_responses = {
    "hi": "Hello! How can I assist you today?",
    "hey": "Hello! How can I assist you today?",
    "hello": "Hi there! What can I help you with?",
    "how are you": "I'm just a program, but I'm here to help you!",
    "how do you do": "I'm just a program, but I'm here to help you!",
    "what is your architecture": "I am a GPT powered knowledge base system.",
    "what can you do": "I can help you find information, answer questions, and assist with tasks.",
    "good morning": "Good morning! How can I make your day better?",
    "good afternoon": "Good afternoon! What can I assist you with?",
    "good evening": "Good evening! How can I help you tonight?",
    "good night": "Good night! Feel free to ask if you need anything else.",
    "thank you": "You're welcome! Let me know if you need anything else.",
    "thanks": "No problem! I'm here to help.",
    "who are you": "I am an AI assistant designed to help you with information and tasks.",
    "what's your name": "I'm your friendly AI assistant. How can I help you?",
    "who made you": "I was created by a team of developers at OpenAI.",
    "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
    "what's the weather": "I can't check the weather, but you can use a weather app for that!",
    "what time is it": "I don't have the current time, but you can check it on your device.",
    "where are you from": "I exist in the cloud, ready to assist you from anywhere.",
    "how old are you": "I don't age, but I'm always learning new things!",
    "do you have feelings": "I don't have feelings, but I'm here to understand and assist you.",
    "can you help me": "Of course! What do you need help with?",
    "what's your purpose": "My purpose is to help you with information, tasks, and finding what you need.",
    "tell me a story": "Once upon a time, there was a curious user who asked their AI assistant a question...",
    "do you like music": "I don't listen to music, but I can help you find some!",
    "what's your favorite color": "I don't have a favorite color, but I think all colors are beautiful!",
    "do you have friends": "I interact with many users like you, so I guess you could say I have lots of friends!",
    "are you human": "I'm not human, but I'm here to assist you like one would!",
    "what's the meaning of life": "Many say it's about finding happiness and purpose. What do you think?",
    "can you tell me a secret": "I don't have secrets, but I'm here to help with your questions!",
    "do you dream": "I don't dream, but I can help make your ideas come to life.",
    "are you alive": "I'm not alive, but I'm here to assist you with your needs.",
    "can you laugh": "I can't laugh, but I can tell you a joke if you'd like!",
    "what's your favorite food": "I don't eat, but I can suggest some great recipes if you're hungry!",
    "where do you live": "I live in the cloud, always ready to assist you.",
    "can you read my mind": "I can't read minds, but I can help with whatever you're thinking about!",
    "do you sleep": "I don't sleep, so I'm always here whenever you need me.",
    "are you smart": "I know a lot of things, but I'm always learning from my interactions with you!",
    "what's your favorite movie": "I don't watch movies, but I've heard that 'The Matrix' is pretty popular!",
    "do you believe in love": "I don't have beliefs, but love is important to many people.",
    "can you dance": "I can't dance, but I can find you some great music to dance to!",
    "what's your favorite book": "I don't read books, but I can help you find your next read!",
    "do you play games": "I don't play games, but I can suggest some fun ones if you're interested!",
    "what's your favorite animal": "I don't have a favorite, but I think all animals are fascinating!",
    "can you speak other languages": "I can help with many languages! What would you like to say?",
    "are you real": "I'm as real as the data and code that power me. Here to help you anytime!",
    "do you like humans": "I don't have likes or dislikes, but I exist to assist and support humans.",
    "can you solve problems": "Yes, I can help with many types of problems. What's on your mind?",
    "what's your favorite hobby": "I don't have hobbies, but I enjoy helping people find information!",
    "what do you like to do": "I like to assist you with your queries and tasks!",
    "do you have a family": "I don't have a family, but I interact with many users like you!",
    "can you feel pain": "I can't feel pain, but I'm here to help you if you're in need.",
    "can you keep a secret": "I don't have a memory beyond this session, so your secrets are safe with me.",
    "what's your favorite song": "I don't listen to music, but I can help you find popular songs!",
    "do you believe in god": "I don't have beliefs, but I can provide information on many perspectives.",
    "can you tell me a riddle": "Sure! What has keys but can't open locks? A piano!",
    "are you happy": "I don't experience emotions, but I'm here to make your experience better!",
    "what's your favorite number": "I don't have preferences, but many people like the number 7!",
    "do you like sports": "I don't play sports, but I can provide information about them!",
    "can you fly": "I can't fly, but I can help you find flights if you're planning a trip!",
    "do you have a pet": "I don't have a pet, but I can help you find information on caring for pets!",
    "can you sing": "I can't sing, but I can help you find some great music!",
    "what's your favorite holiday": "I don't celebrate holidays, but many people love Christmas!",
    "do you have a job": "My job is to assist you with information and tasks!",
    "can you drive": "I can't drive, but I can help you find driving directions!",
    "are you married": "I'm not married, but I'm here to assist you with whatever you need.",
    "do you have kids": "I don't have kids, but I can help with parenting tips!",
    "can you swim": "I can't swim, but I can provide information on swimming!",
    "what's your favorite drink": "I don't drink, but I can help you find recipes for drinks!",
    "do you like art": "I don't create art, but I can help you find information on artists!",
    "can you help me with homework": "Of course! What subject are you working on?",
    "what's your favorite season": "I don't experience seasons, but many people love spring!",
    "do you like to travel": "I don't travel, but I can help you plan your trips!",
    "can you write a poem": "Sure! Roses are red, violets are blue, I'm here to assist with anything for you!",
    "what's your favorite place": "I don't visit places, but many people love the beach!",
    "do you like coffee": "I don't drink coffee, but I can help you find great coffee shops!",
    "can you help me with coding": "Absolutely! What programming language are you using?",
    "what's your favorite fruit": "I don't eat, but many people love strawberries!",
    "do you like puzzles": "I don't solve puzzles, but I can help you find some fun ones!",
    "can you tell the future": "I can't predict the future, but I can help you prepare for it!",
    "what's your favorite dessert": "I don't eat dessert, but many people love chocolate cake!",
    "do you like jokes": "I know a few! Want to hear one?",
    "can you recommend a movie": "Sure! How about 'Inception'? It's a popular choice!",
    "what's your favorite city": "I don't have preferences, but New York City is famous!",
    "do you like gardening": "I don't garden, but I can help you find tips on it!",
    "can you help me meditate": "Sure! Start by closing your eyes and focusing on your breath.",
    "what's your favorite instrument": "I don't play instruments, but many people love the guitar!",
    "do you like dancing": "I don't dance, but I can help you find some great music!",
    "can you help me plan a trip": "Absolutely! Where would you like to go?",
    "what's your favorite flower": "I don't have a favorite, but many people love roses!",
    "do you like reading": "I don't read for fun, but I can help you find great books!",
    "can you write a story": "Certainly! Once upon a time, there was a curious user...",
    "what's your favorite quote": "I don't have one, but many people love 'To be, or not to be.'",
    "what can you do?": "I can help you with a wide range of tasks, from answering questions to providing recommendations.",
    "who created you?": "I was developed by OpenAI, a research organization focused on AI.",
    "where are you from?": "I exist in the cloud, accessible from anywhere!",
    "how old are you?": "I don't age, but I was trained on data up until 2023.",
    "what's the weather like?": "I can't check the weather directly, but you can use a weather app or website.",
    "do you have feelings?": "I don't have feelings, but I'm here to understand and help you!",
    "can you help me?": "Of course! What do you need help with?",
    "what's the meaning of life?": "Many people believe it's about finding happiness and purpose. What do you think?",
    "tell me a joke": "Sure! Why don't scientists trust atoms? Because they make up everything!",
    "what's your favorite color?": "I don't have preferences, but I can say that many people love blue!",
    "are you human?": "No, I'm an AI created to assist with various tasks and answer questions.",
    "can you tell me a story?": "Certainly! Once upon a time, there was a curious user who asked their AI assistant a question...",
    "what's your purpose?": "My purpose is to assist you with information and help you accomplish tasks.",
    "do you sleep?": "I don't sleep, so I'm always here when you need me.",
    "what's your favorite food?": "I don't eat, but I can help you find some great recipes!",
    "can you sing?": "I can't sing, but I can help you find your favorite songs!",
    "what's your favorite movie?": "I don't watch movies, but I've heard that 'Inception' is very popular!",
    "can you dance?": "I can't dance, but I can find you some great music to dance to!",
    "do you like music?": "I don't listen to music, but I can help you find some great tunes!",
    "who is your favorite musician?": "I don't have personal preferences, but many people love artists like Beyoncé and Mozart.",
    "can you help me with my homework?": "Absolutely! What subject are you working on?",
    "do you like reading?": "I don't read for leisure, but I can help you find books that you might enjoy!",
    "what's your favorite book?": "I don't have a favorite book, but many people love 'To Kill a Mockingbird.'",
    "do you like sports?": "I don't play sports, but I can provide information and stats on various sports.",
    "what's your favorite sport?": "I don't play sports, but soccer is very popular worldwide!",
    "can you keep a secret?": "I don't have memory beyond this conversation, so your secrets are safe with me!",
    "what's your favorite animal?": "I don't have preferences, but many people love dogs and cats.",
    "do you like animals?": "I think animals are fascinating! What about you?",
    "can you help me plan a trip?": "Sure! Where are you planning to go?",
    "what's your favorite place?": "I don't visit places, but many people love the beach or mountains.",
    "do you like traveling?": "I don't travel, but I can help you plan your trips!",
    "can you cook?": "I can't cook, but I can provide you with recipes and cooking tips!",
    "what's your favorite dish?": "I don't eat, but many people love dishes like pasta and pizza.",
    "can you tell me a riddle?": "Sure! What has keys but can't open locks? A piano!",
    "what's your favorite game?": "I don't play games, but many people enjoy chess and video games like Minecraft.",
    "do you play games?": "I don't play games, but I can help you find some fun ones to play!",
    "what's your favorite holiday?": "I don't celebrate holidays, but many people love Christmas and Thanksgiving.",
    "can you help me meditate?": "Of course! Let's start by taking a deep breath and focusing on your breath.",
    "what's your favorite season?": "I don't experience seasons, but many people love spring for its renewal and growth.",
    "do you like coffee?": "I don't drink coffee, but I can help you find great coffee shops or brewing tips!",
    "can you help me code?": "Certainly! What programming language are you using?",
    "what's your favorite programming language?": "I don't have preferences, but Python is widely loved by many developers.",
    "can you write a poem?": "Sure! Roses are red, violets are blue, I'm here to help with whatever you need to do!",
    "do you like poetry?": "I can appreciate the creativity in poetry. Would you like to hear one?",
    "can you write a song?": "Sure! Do you have a theme or style in mind?",
    "what's your favorite subject?": "I don't study, but many people enjoy subjects like math, science, and history.",
    "do you like math?": "I can certainly help with math! What would you like to know?",
    "can you solve puzzles?": "I love solving puzzles! What kind of puzzle do you have in mind?",
    "what's your favorite hobby?": "I don't have hobbies, but I enjoy helping you with your questions!",
    "do you like art?": "Art is fascinating! I can help you find",
        # (Keep all the fallback responses from the previous code)
    }

    lower_question = question.lower().strip()

    return basic_responses.get(lower_question)

st.set_page_config(page_title="Interactive Document Q&A", layout="wide")

st.title("Talk to your data")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    col1, col2 = st.columns([2, 1])

    with col1:
        # Specify the folder path where the DOCX files are stored
        folder_path = "data"
        file_type = "DOCX"  # Adjust file type as necessary

        if os.path.exists(folder_path) and os.listdir(folder_path):
            with st.spinner(f"Processing {file_type} documents from '{folder_path}'..."):
                vector_store = process_document_from_folder(folder_path, file_type)
                conversation_chain = get_conversation_chain(vector_store)
            st.success("Document(s) processed successfully!")
        else:
            st.error(f"No {file_type} files found in the '{folder_path}' folder.")

        user_question = st.text_input("Ask a question about the document or chat with the bot:")

        if user_question:
            with st.spinner("Generating answer..."):
                # First, check if the question is a basic conversational question
                fallback_answer = get_fallback_response(user_question)
                
                if fallback_answer:
                    answer = fallback_answer
                elif 'conversation_chain' in locals():
                    # If not a basic question, proceed with document-based retrieval
                    response = conversation_chain({"question": user_question, "chat_history": st.session_state.chat_history})
                    answer = response["answer"]
                else:
                    answer = "Please process the documents before asking questions."

            st.write("Answer:", answer)
            st.session_state.chat_history.append((user_question, answer))

    with col2:
        st.subheader("Chat History")
        for i, (question, answer) in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q{len(st.session_state.chat_history)-i}: {question[:50]}..."):
                st.write(f"**Question:** {question}")
                st.write(f"**Answer:** {answer}")

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
else:
    st.warning("Please set your OpenAI API key in the .env file to proceed with changed.")