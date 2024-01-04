import os
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, ConversationChain 
from langchain.document_loaders.csv_loader import CSVLoader 
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import tempfile

user_api_key = st.sidebar.text_input(
label="Enter API Key", placeholder="Enter API Key")
uploaded_file=""

if not user_api_key:
    st.error('Please provide your API key before proceeding.')
else:
    uploaded_file = st.sidebar.file_uploader("upload", type="pdf")
    data=None

    os.environ["OPENAI_API_KEY"] = user_api_key

    if uploaded_file:

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file: 
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        #loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={ 'delimiter': ','})
        #data = loader.load()
        loader = PyPDFLoader(file_path=tmp_file_path)
        data = loader.load_and_split()

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(data, embeddings)

        chain = ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(temperature=0.0,model_name='gpt-4-1106-preview'), 
        retriever=vectorstore.as_retriever())   
    else:

        llm = ChatOpenAI(temperature=0.0,model_name='gpt-4-1106-preview')
        no_rag_chain = ConversationChain(
            llm=llm, verbose=True,
        )
        
def conversational_chat(query, uploaded_file):
    if uploaded_file:
        

        result = chain({"question": query,
        "chat_history": st.session_state['history']}) 
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    else:
        result= no_rag_chain({"input": query})
        print(result)
        return result["response"]

if 'history' not in st.session_state: 
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    if uploaded_file:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + ":hugging_face:"]
    else:
        st.session_state['generated'] = ["How can I help you today? "]
if 'past' not in st.session_state: st.session_state['past'] = ["Hey"]

#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):

        user_input = st.text_input("Query:", placeholder="Please put your request here", key='input')
        submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    output = conversational_chat(user_input, uploaded_file)   

    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

