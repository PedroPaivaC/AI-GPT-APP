# Imports
from All_Pinecone_Admin import query
# Langchain Imports
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
# Credentials Import
from credentials_call import credential
# Warning Mitigation Import
import warnings

# FIXME: Mitigate error message!
warnings.filterwarnings('ignore', category=DeprecationWarning,
                        message='The method `Chain.run` was deprecated in langchain 0.1.0'
                                ' and will be removed in 0.3.0. Use invoke instead.')


def question_answer(question, temperature=0, chain_type='stuff'):

    # Returns documents from Pinecone Vector Database based on Semantic Search from {Question}
    document_input = query(question)

    # Defines LLM to be used
    large_language_model = OpenAI(api_key=credential('OPENAI_API_KEY'), temperature=temperature)

    # Creates Question and Answer chain (Langchain -> LLM)
    chain = load_qa_chain(large_language_model, chain_type=chain_type)

    response = chain.run(input_documents=document_input, question=question)

    return response
