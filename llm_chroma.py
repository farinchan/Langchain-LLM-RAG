
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA  


import os

from dotenv import load_dotenv
load_dotenv()

def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

doc=read_doc('documents/')
print('Number of documents: ',len(doc))

def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    docs=text_splitter.split_documents(docs)
    return docs

documents=chunk_data(docs=doc)
print('Number of chunks: ',len(documents))

embeddings=OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
print('Embeddings loaded : ', embeddings)

vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

llm = OpenAI()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)


while True:
    print("\n Chat with me! Type 'exit' to quit.\n")
    query = input("tanyakan tentang skripsi saya: ")
    if query == "exit":
        break
    response_qa = qa.invoke(query)
    
    print("QA response: ", response_qa)

