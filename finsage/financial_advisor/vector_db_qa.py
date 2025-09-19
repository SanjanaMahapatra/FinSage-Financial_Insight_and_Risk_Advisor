import pandas as pd
import numpy as np

from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain_openai.chat_models import ChatOpenAI

import os
import json
from dotenv import load_dotenv
import openai
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model: str = "text-embedding-3-large"

# initialize OPENAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)


# loading the bank of baroda website data, this will act as the knowledge base
def load_docs(file_path):
    docs = []
    with open(file_path, 'r') as jsonl_data:
        for line in jsonl_data:
            data = json.loads(line)
            obj = Document(**data)
            docs.append(obj)

    return docs

docs = load_docs('./bob_website_data.jsonl')


docs_length = []
for i in range(len(docs)):
    docs_length.append(len(docs[i].page_content))

print(f'doc lengths\nmin: {min(docs_length)} \navg.: {round(np.average(docs_length), 1)} \nmax: {max(docs_length)}')

# chunking the docks of the doc_length received from bob website data 

chunk_size = 1000
chunk_overlap = 200

def chunk_docs(doc, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len)
    chunks = text_splitter.create_documents(texts=[doc.page_content], metadatas=[{'source': doc.metadata['source']}])
    return chunks

chunked_docs = []

for i in docs:
    chunked_docs.append(chunk_docs(i, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
flattened_chunked_docs = [doc for docs in chunked_docs for doc in docs]

print("Print the flattened chunked docs ==> ", flattened_chunked_docs)

# initialize the OpenAI Embeddings, and storing in the faiss vector db

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

vector_db = FAISS.from_documents(flattened_chunked_docs, embedding=embeddings)

vector_db.save_local("faiss_index_folder")

query_docs = vector_db.similarity_search(
    query= 'What is the data about? Explain in brief',
    k = 10,
    search_type='similarity'
)

llm = ChatOpenAI(temperature=0.1,model="gpt-4o-mini", max_retries=2, n=3)

prompt_template = """You are an expert financial advisor from Bank of Baroda, assisting users with specific Bank of Baroda products based on their financial data and user profile.
Context:
{context}
Chat history:
{chat_history}
Query: {question}
"""

prompt = PromptTemplate(template = prompt_template, input_variables= ["chat_history", "question"])

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(),
    combine_docs_chain_kwargs = {"prompt": prompt},
    return_source_documents = True,
    verbose=True
)

chat_history = []


def query_vector_db(query):
    qa_result = qa({
        "question": query,
        "chat_history": chat_history,
    })
    
    print(" qa_result ==> ", qa_result);

    sources = []
    for i in qa_result['source_documents']:
        sources.append(i.metadata['source'])
    
    qa_result = {'answer': qa_result['answer'], "sources": sources}
    
    return qa_result


