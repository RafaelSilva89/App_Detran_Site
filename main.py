# pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb

#from langchain_core.messages              import AIMessage, HumanMessage
#from langchain_community.document_loaders import WebBaseLoader
#from langchain.text_splitter              import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores     import Chroma
#from langchain_openai                     import OpenAIEmbeddings, ChatOpenAI
#from langchain.llms                       import OpenAI
#from langchain_core.prompts               import ChatPromptTemplate, MessagesPlaceholder
#from langchain.chains                     import create_history_aware_retriever, create_retrieval_chain
#from langchain.chains.combine_documents   import create_stuff_documents_chain
#import openai

import streamlit as st
from dotenv                               import load_dotenv
from langchain.chat_models                import ChatOpenAI
from langchain.embeddings                 import OpenAIEmbeddings
from langchain_core.messages              import AIMessage, HumanMessage, SystemMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter              import RecursiveCharacterTextSplitter
from langchain_community.vectorstores     import Chroma
from langchain_core.prompts               import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains                     import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents   import create_stuff_documents_chain
from htmlTemplates                        import css, bot_template, user_template

load_dotenv()

def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200, length_function=len)
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    #llm = ChatOpenAI()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.2, max_tokens=1000)

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.2, max_tokens=1000)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response['answer']

# app config
st.set_page_config(page_title="Chat with websites", page_icon=":globe_with_meridians:")
st.title("Chat with websites :globe_with_meridians:")

# sidebar
with st.sidebar:
    st.header("Settings")
    #website_url = st.text_input("Website URL")
    website_url = st.text_area("Website URL", height=100)
    st.markdown('''
            - [Streamlit](https://streamlit.io/)
            - [LangChain](https://python.langchain.com/)
            - [OpenAI](https://platform.openai.com/docs/models) LLM Model
            ''')
    # st.write('Also check out my portfolio for amazing content [Rafael Silva](https://rafaelsilva89.github.io/portfolioProjetos/#)')

if website_url is None or website_url == "":
    st.info("Please enter a website URL")


else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

        # user input
    user_query = st.chat_input("Ask a question about your documents...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:
        st.write(css, unsafe_allow_html=True)
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                #st.write(message.content)
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                #st.write(message.content)
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)