from lang_funcs import *
from langchain.llms import Ollama
from langchain import PromptTemplate

# Loading orca-mini from Ollama
llm = Ollama(model="orca-mini", temperature=0)

# Loading the Embedding Model
embed = load_embedding_model(model_path="all-MiniLM-L6-v2")

def load_pdf_data(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()
# loading and splitting the documents
docs = load_pdf_data(file_path="data/SomilJain_CV_21_24.pdf")
documents = split_docs(documents=docs)

# creating vectorstore
vectorstore = create_embeddings(documents, embed)

# converting vectorstore to a retriever
retriever = vectorstore.as_retriever()

# Creating the prompt from the template which we created before
prompt = PromptTemplate.from_template(template)

# Creating the chain
chain = load_qa_chain(retriever, llm, prompt)
# Running the chain with a sample query
# query = 
# response = chain.run(query)
# print(response)
get_response("who is somil jain",chain)