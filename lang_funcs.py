from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
#from langchain.document_loaders import PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import textwrap
"""_______________________________________________________________"""
# # Load the PDF document
# pdf_loader = PDFLoader(file_path='path/to/your/document.pdf')
# documents = pdf_loader.load()

# # Split the text into manageable chunks
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents(documents)

# # Load the OCRa Mini model embeddings
# embeddings = OllamaEmbeddings(model='ocra-mini')

# # Create a FAISS vector store from the text chunks
# vector_store = FAISS.from_documents(texts, embeddings)

# # Initialize the Ollama LLM
# llm = Ollama(model='ocra-mini')

# # Create a RetrievalQA chain
# qa_chain = RetrievalQA(llm=llm, retriever=vector_store.as_retriever())

# # Function to search the PDF
# def search_pdf(query):
#     result = qa_chain.run(query)
#     return result

# # Example usage
# query = "Your search query here"
# result = search_pdf(query)
# print(result)

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents=documents)
    return texts

def load_embedding_model(model_path,normalize_embeddings=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings':normalize_embeddings} #cosine similarity
    )
def create_embeddings(chunks,embedding_model,storing_path="vectorstore"):
    vectorstore=FAISS.from_documents(chunks,embedding_model)
    vectorstore.save_local(storing_path)
    return vectorstore

prompt = """
### System:
You are an AI Assistant that follows instructions extreamly well. \
Help as much as you can.

### User:
{prompt}

### Response:

"""

template = """
### System:
You are an respectful and honest assistant. You have to answer the user's \
questions using only the context provided to you. If you don't know the answer, \
just say you don't know. Don't try to make up an answer.

### Context:
{context}

### User:
{question}

### Response:
"""

# Creating the chain for Question Answering
def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, # here we are using the vectorstore as a retriever
        chain_type="stuff",
        return_source_documents=True, # including source documents in output
        chain_type_kwargs={'prompt': prompt} # customizing the prompt
    )

def get_response(query, chain):
    # Getting response from chain
    response = chain({'query': query})
    
    # Wrapping the text for better output in Jupyter Notebook
    wrapped_text = textwrap.fill(response['result'], width=100)
    print(wrapped_text)