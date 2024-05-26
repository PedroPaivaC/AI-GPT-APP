from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import pandas as pd
import requests
import os

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

load_dotenv(dotenv_path='credentials.env')
pc = Pinecone(api_key=os.getenv('API_KEY_PINECONE'))
index_name = os.getenv('PINECONE_INDEX_NAME')

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)


def embed_and_store(file_path, sort: str,
                     chunk_size=1000, chunk_overlap=0):

    '''Embeds the provided content using (all-MiniLM-L6-v2) embedding model and
    appends Pinecone Vector Database with the converted vectors.'''

    sort = sort.lower()

    loader = None

    if sort == 'pdf':
        loader = PyPDFLoader(file_path)
    elif sort == 'txt':
        loader = TextLoader(file_path)

    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # Use named parameters
    docs = text_splitter.split_documents(documents)

    doc_texts = [doc.page_content for doc in docs]
    doc_embeddings = embedding_model.encode(doc_texts)

    vectors = [(str(i), doc_embeddings[i].tolist()) for i in range(len(doc_embeddings))]
    index.upsert(vectors)


def query(question, top_k=10):

    '''Performs semantic search on Pinecone Vector Database
    and returns 'top_k' closest neighbors.'''

    query_embedding = embedding_model.encode([question])[0].tolist()
    results = index.query(vector=query_embedding, top_k=top_k)

    return results


def empty_vector_space():

    '''Empties Pinecone Vector Space.
    Should be used at the end of applications or to separate them.'''

    pc = Pinecone(api_key=os.getenv('API_KEY_PINECONE'))

    if index_name in pc.list_indexes().names():
        index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))

        index.delete(delete_all=True)
        print(f'All Vectors Have Been Deleted!')
    else:
        print(f'The Provided Index_Name Does Not Exist!')


if __name__ == '__main__':

    pass
