from dotenv import load_dotenv
import os


def credential(cred_name: str):
    load_dotenv(dotenv_path='credentials.env')
    return os.getenv(cred_name)
