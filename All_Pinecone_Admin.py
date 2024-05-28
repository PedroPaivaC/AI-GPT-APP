# Langchain Imports (TextLoader: .md, .txt - PyPDFLoader: .PDF - YoutubeLoader: Hyperlink)
from langchain_community.document_loaders import TextLoader, PyPDFLoader, YoutubeLoader
from langchain_text_splitters import CharacterTextSplitter  # Document Chunk Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # Embedding Format -> Document Format
# Sentence Transformer Import
from sentence_transformers import SentenceTransformer
# Pinecone Import
from pinecone import Pinecone, ServerlessSpec
# Credentials Import
from credentials_call import credential
# Warning Mitigation Import
import warnings
# General Imports
import pandas as pd
import time
import os

# FIXME: Mitigate error message!
warnings.filterwarnings('ignore', category=FutureWarning,
                        message='`resume_download` is deprecated and will be removed in version 1.0.0. '
                                'Downloads always resume when possible. If you want to force a new download, '
                                'use `force_download=True`.')

model_name = 'all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(model_name)

pc = Pinecone(api_key=credential('API_KEY_PINECONE'))
index_name = credential('PINECONE_INDEX_NAME')
index = pc.Index(index_name)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )


def embed_and_store(file_path, chunk_size=1000, chunk_overlap=0, language='en', translation='en'):

    '''Embeds the provided content using (all-MiniLM-L6-v2) embedding model and
    appends Pinecone Vector Database with the converted vectors.'''

    split_path = file_path.split('.')
    sort = split_path[-1]
    loader = None

    # Ensures the same or different embeddings are not appended to the
    # Vector Database at the same time - Break
    while True:

        if 'youtube' in split_path:

            loader = YoutubeLoader.from_youtube_url(file_path, add_video_info=True,
                                                    language=[language], translation=translation)

            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            texts = text_splitter.split_documents(documents)

            vectors = []
            for i, doc in enumerate(texts):
                doc_text = doc.page_content
                doc_embedding = embedding_model.encode(doc_text)
                vectors.append({
                    'id': f'youtube_{i}_{int(time.time())}',
                    'values': doc_embedding.tolist(),
                    'metadata': {'text': doc_text}
                })

            index.upsert(vectors)
            time.sleep(3)

            break

        elif sort == 'pdf':
            loader = PyPDFLoader(file_path)
        elif sort == 'txt':
            loader = TextLoader(file_path)
        elif sort == 'md':
            loader = TextLoader(file_path)

        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)

        vectors = []
        for i, doc in enumerate(docs):
            doc_text = doc.page_content
            doc_embedding = embedding_model.encode(doc_text)
            vectors.append({
                'id': f'{sort}_{i}_{int(time.time())}',
                'values': doc_embedding.tolist(),
                'metadata': {'text': doc_text}
            })

        index.upsert(vectors)
        time.sleep(3)


def query(question, top_k=3):

    '''Performs semantic search on Pinecone Vector Database
    and returns 'top_k' closest neighbors.'''

    model = SentenceTransformer(model_name)

    embedded_query = model.encode(question).tolist()
    results = index.query(vector=embedded_query, top_k=top_k, include_metadata=True)

    documents = []
    for match in results['matches']:
        clean_text = match['metadata']['text'].replace('\n', ' ').replace('  ', ' ').strip()
        doc = Document(page_content=clean_text, metadata={'score': match['score']})
        documents.append(doc)

    return documents


def empty_vector_space():

    '''Empties Pinecone Vector Space.
    Should be used at the end of applications or to separate them.'''

    pinecone_vd = Pinecone(api_key=credential('API_KEY_PINECONE'))

    if index_name in pinecone_vd.list_indexes().names():

        index.delete(delete_all=True)
        print(f'All Vectors Have Been Deleted!')
    else:
        print(f'The Provided Index_Name Does Not Exist!')
