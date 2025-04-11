import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import time
import pickle
from dotenv import load_dotenv
load_dotenv()

st.title("News Research Tool📈")
st.sidebar.title("News Article URLS")
urls = []
for i in range(3):
  url =  st.sidebar.text_input(f"URL {i+1}")
  urls.append(url)

process_url_clicked = st.sidebar.button("Process URLS")
file_path = "faiss_store_openai.pkl"

main_placefolder = st.empty()

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading...Started...")
    data = loader.load()
    print(len(data))
    #SPLIT DATA
    text_splitter = RecursiveCharacterTextSplitter(
       separators = ['\n\n', '\n', '.', ','],
       chunk_size = 1000
    )

    main_placefolder.text("Data Loading...Completed...")
    main_placefolder.text("Text Splitter...Started...")

    docs = text_splitter.split_documents(data)


    #Create embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"  
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorindex_openai = FAISS.from_documents(docs, embeddings)

    main_placefolder.text("Embedding vector started Building....")
    time.sleep(2)

    with open(file_path, "wb") as f:
       pickle.dump(vectorindex_openai, f)