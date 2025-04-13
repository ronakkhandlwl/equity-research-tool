import streamlit as st
import pickle, time, os
from langchain.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=os.getenv("HUGGING_FACE_API_KEY")
)



st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    urls.append(st.sidebar.text_input(f"URL {i+1}"))

process_url_clicked = st.sidebar.button("Process URLs")
filepath = 'faiss_store__openai.pkl'

main_placefolder = st.empty()

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Loading Data...")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.'],
        chunk_size=1000
        )
    main_placefolder.text("Splitting Data...")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save it to FAISS indexes
    embeddings = HuggingFaceEmbeddings()
    main_placefolder.text("Embedding Vector...")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    # Save the FAISS index to a pickle file
    time.sleep(2)
    with open(filepath, "wb") as f:
        pickle.dump(vectorstore_openai, f)


query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.subheader(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
