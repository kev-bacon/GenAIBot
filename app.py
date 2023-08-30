import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# , HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
# from langchain.llms import HuggingFaceHub
# from langchain.document_loaders import TextLoader
# import pinecone 
# from langchain.vectorstores import Pinecone
import os
import pandas as pd
import glob
from pptx import Presentation



# ##CHAT GPT HAS TO be 1536 embedding dimensions, must check for Google
# def init_pinecone(): 
#     pinecone.init(
#         api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
#         environment=os.getenv("PINECONE_ENV"),  # next to api key in console
#     )
#     index_name = "dackai"
#     if index_name not in pinecone.list_indexes(): 
#         pinecone.create_index(name=index_name, dimension=1536, metric="cosine")
#         print(f"create a new index {index_name}")
#     else:
#         print(f"{index_name} index exist. don't create this again")
#     return index_name

# def init_pinecone():
#     env_file_loader = load_dotenv(find_dotenv())
#     pinecone.init(
#         api_key=os.environ['PINECONE_API_KEY'],  # find at app.pinecone.io
#         environment=os.environ['PINECONE_ENV'],  # next to api key in console
#     )
#     index_name = "dackai"
#     if index_name not in pinecone.list_indexes():
#         pinecone.create_index(name=index_name, dimension=1536, metric="cosine")
#         print(f"create a new index {index_name}")
#     else:
#         print(f"{index_name} index exist. don't create this again")
#     return index_name



#  puts all pdf text into a variable and returns it, pretty chill  
def get_pdf_text(pdf_docs):
    print(pdf_docs)
    text = ""
    for pdf in pdf_docs:
        print(pdf)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def csv_to_pd(documents):
    #temp location for csv
    os.chdir('/temp-data')
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    dfs = []
    for csv in documents:
        df = pd.read_csv(csv)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def pptx_to_text(documents):
    text = ''
    for eachfile in documents:
        prs = Presentation(eachfile)
        print(eachfile)
        print("----------------------")
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    print(shape.text)
                    text += shape.text
    return text

def all_files(documents):
    print(documents)
    for eachfile in documents:
        if eachfile.name.endswith('.csv'):
            text = csv_to_pd(documents)
            print(eachfile.name + '  here')
        elif eachfile.name.endswith('.pdf'):
            text = get_pdf_text(documents)
            print(eachfile.name + '  here')
        elif eachfile.name.endswith('.pptx'):
            text = pptx_to_text(documents)
            print(eachfile.name + '  here')
    return(text)

#uses langchain function to split text -> so that it can be used for embedding
def get_text_chunks(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=5000,
        chunk_overlap=500,
        length_function=len
    )
    chunks = text_splitter.split_text(documents)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl") #can switch to instructor if stronger GPU, but im broke 
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # index_name = init_pinecone()
    # vectorstore = Pinecone.from_documents(text_chunks, embedding= embeddings, index_name = index_name)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation is None:
        st.write(bot_template.replace("{{MSG}}","I do not have enough context, please upload a relevant PDF for me to learn"), unsafe_allow_html=True) 
    elif user_question: 
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Created by the 3 musketeers (kevin.b.nguyen, callum.linnegan, arushi.tejpal)")
        file_type = st.radio(
            "What file type would you like to upload?",
            ["PDF","CSV","PPTX","all"]
        )
        documents = st.file_uploader(
            "Upload your files here and click on 'Process'", accept_multiple_files=True)
        #print(documents)
        if st.button("Process"):
            with st.spinner("Processing"):    ##Show that it is running/not frozen 
                # get pdf text
                # if file_type == 'PDF':
                #     raw_text = get_pdf_text(documents)
                # elif file_type == 'CSV':
                #     raw_text = csv_to_pd(documents)
                # elif file_type == 'PPTX':
                #     raw_text = pptx_to_text(documents)
                if file_type == 'all':
                    raw_text = all_files(documents)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
