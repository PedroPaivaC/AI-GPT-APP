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

    while True:

        print()

        query = input('Question: ')

        answer = question_answer(question=query, temperature=0.5)

        print(answer)

        time.sleep(3)
