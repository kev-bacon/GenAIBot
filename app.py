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
#     pinecone.init(
#         api_key='PINECONE_API_KEY',  # find at app.pinecone.io
#         environment='PINECONE_ENV'  # next to api key in console
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
        # print(eachfile)
        # print("----------------------")
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    print(shape.text)
                    text += shape.text
    return text

def all_files(documents):
    text = ''
    # print('ALL_FILES', text)
    for eachfile in documents:
        # print('eachfile',eachfile.name)
        if eachfile.name.endswith('.csv'):
            text += csv_to_pd([eachfile])
            # print(eachfile.name + '  here')
        elif eachfile.name.endswith('.pdf'):
            text += get_pdf_text([eachfile])
            # print(eachfile.name + '  here')
        elif eachfile.name.endswith('.pptx'):
            text += pptx_to_text([eachfile])
            # print(eachfile.name + '  here')
    return(text)

#uses langchain function to split text -> so that it can be used for embedding
def get_text_chunks(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
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
                       page_icon=":books:",
                       layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.image("https://1000logos.net/wp-content/uploads/2021/04/Accenture-logo-500x281.png", width=100)
    st.header("Ask questions about your documents")
    user_question = st.text_input("Type your question below:")
    if user_question and st.session_state.conversation is None:
        st.write(bot_template.replace("{{MSG}}","I do not have enough context, please upload a relevant PDF for me to learn"), unsafe_allow_html=True) 
    elif user_question: 
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Created by the 3 musketeers (kevin.b.nguyen, callum.linnegan, arushi.tejpal)")
        documents = st.file_uploader(
            "Upload your files here and click on 'Process'", accept_multiple_files=True)
        #print(documents)
        if st.button("Process"):
            with st.spinner("Processing"):    ##Show that it is running/not frozen \
                raw_text = all_files(documents)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
            st.success("Files finished processing", icon="✅")
        LLM_List = ('GPT-3.5', 'Llama 2.0', 'PaLM')
        option = st.selectbox(
        'Which LLM would you like to use',
        LLM_List)
        if option in LLM_List:
            add_slider = st.sidebar.slider(
                'Temperature',
                0.0, 1.0, 0.8, 0.05
                )
        if option == 'GPT-3.5':
            add_slider = st.sidebar.slider(
                'Top P',
                0.0, 1.0, 1.0, 0.05
                )
            add_slider = st.sidebar.slider(
                'Frequency Penalty',
                0.0, 1.0, 0.0, 0.05
                )

        st.write('You selected:', option)
        


if __name__ == '__main__':
    main()

