
import pandas as pd
import pinecone


from langchain_pinecone import PineconeVectorStore
from langchain.agents import Tool
from langchain.agents import create_json_chat_agent
from langchain import hub
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from aws_secrets_initialization import PINECONE_API_KEY, INDEX_NAME,COHERE_API_KEY, embeddings, llm

# Constants and configuration

def initialize_vector_store(index_name):
    """
    Initializes and returns a Pinecone Vector Store with specific index and embeddings.
    """
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    text_field = "text"
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index, embeddings, text_field)
    return vector_store

def retrieve_documents(query):
    """
    Retrieves documents relevant to a query using a vector store and contextual compression.
    """
    vector_store = initialize_vector_store(INDEX_NAME)
    retriever = vector_store.as_retriever(search_kwargs={'k': 100})
    compressor = CohereRerank(top_n=20, model = 'rerank-english-v2.0', cohere_api_key=COHERE_API_KEY)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    documents = compression_retriever.invoke(query)
    return documents

# Tool setup for LangChain
knowledge_base_tool = Tool(
    name='Knowledge Base',
    func=retrieve_documents,
    description='Use this tool to answer questions about Federal Student Aid Handbook or FAFSA, providing more information about the topic.'
)

# Chat agent configuration and initialization
tools = [knowledge_base_tool]
chat_prompt = hub.pull("hwchase17/react-chat-json")
chat_agent = create_json_chat_agent(llm=llm, tools=tools, prompt=chat_prompt)
