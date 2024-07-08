
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.llms import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain


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

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "langchainvector"
namespace = "wondervector5000"

docsearch = PineconeVectorStore.from_documents(
    documents=documents,
    index_name=index_name,
    embedding=embeddings, 
    namespace = namespace

)
    
from langchain.chains import RetrievalQA  

llm = OpenAI()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)


while True:
    print("\n Chat with me! Type 'exit' to quit.\n")
    query = input("tanyakan tentang skripsi saya: ")
    if query == "exit":
        break
    response_qa = qa.invoke(query)
    
    print("QA response: ", response_qa)

