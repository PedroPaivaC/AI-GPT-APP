# Imports
from All_Pinecone_Admin import embed_and_store, query, empty_vector_space
from OpenAI_API import question_answer
# Langchain Imports
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
# Credentials Import
from credentials_call import credential
# openai Import
import openai
# General Imports
import time
import os

os.environ["TOKENIZERS_PARALLELISM"] = 'false'

if __name__ == '__main__':

    # embed_and_store('https://www.youtube.com/watch?v=8Ym7f1an0Rs&list=PL6LH0ngwf3Ht1l7yOr7zsWB77ywwMsoUb')
    # embed_and_store('Extended Essay copy.pdf')

    # empty_vector_space()

    while True:

        question = input('You: ')

        response = question_answer(question=question, temperature=0.3)

        print(f'AI:{response}')

