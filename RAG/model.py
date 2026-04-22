from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

load_dotenv()

model = ChatOpenAI(
    model="stepfun/step-3.5-flash",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
)

import os

path = os.getcwd()

# loading documents
loader = DirectoryLoader(path=path, glob="*.pdf", loader_cls=PyPDFLoader)

document = loader.load()


#splitting the document into chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks=text_splitter.split_documents(documents=document)

# print(len(chunks))
# print(chunks[6].page_content)
embedding=OpenAIEmbeddings()
vector_store=Chroma.from_documents(embedding=embedding,
documents=chunks)


retriver=vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={"k":5,"lambda_mult":0.5},

)

def get_content(query):
    contents=retriver.invoke(query)
    full_content=""
    for content in contents:
        full_content+=content.page_content
    
    return full_content





prompt=PromptTemplate(
    template="accordint to question:{question} and content:{content} you need to analysis and give precise answer",
    input_variables=['question','content']
)

parser=StrOutputParser()

chain=prompt|model|parser


from model import chain

user_input=input("User:")

response=chain.invoke({"question":user_input,"content":get_content(user_input)})

print("AI",response)




