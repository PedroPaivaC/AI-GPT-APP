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


def question_answer(question, top_k=5, temperature=0, chain_type='stuff'):

    """Returns modified answers from All_Pinecone_Admin.query() using a
        Large Language Model.

    :param question: Query to have semantic search performed on the respective
        Vector Database.
    :type question: str
    :param top_k: Number of closest neighbors to be returned upon semantic search.
        Defaults to 5.
    :type top_k: int, optional
    :param temperature: Level of randomness and 'creativity' allowed for the LLM.
       (0: Lowest, 1: Highest). For the same question and data, 'temperature=0'
       shall return the same response. Defaults to 0.
    :type temperature: float
    :param chain_type: ???
    :type chain_type: str, optional

       """

    # Returns documents from Pinecone Vector Database based on Semantic Search from {Question}
    document_input = query(question, top_k=top_k)

    # Defines LLM to be used
    large_language_model = OpenAI(api_key=credential('OPENAI_API_KEY'), temperature=temperature)

    # Creates Question and Answer chain (Langchain -> LLM)
    chain = load_qa_chain(large_language_model, chain_type=chain_type)

    response = chain.run(input_documents=document_input, question=question)

    return response
