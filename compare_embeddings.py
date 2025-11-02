from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {len(vector)} dimensions")

if __name__ == "__main__":
    main()
