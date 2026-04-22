from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

# Create documents for different IPL players
documents = [
    Document(
        page_content="Virat Kohli is a top-order batsman known for his consistency and aggressive batting style.",
        metadata={
            "name": "Virat Kohli",
            "team": "Royal Challengers Bangalore",
            "role": "Batsman",
            "country": "India",
        },
    ),
    Document(
        page_content="MS Dhoni is a legendary wicketkeeper-batsman and one of the most successful captains in IPL history.",
        metadata={
            "name": "MS Dhoni",
            "team": "Chennai Super Kings",
            "role": "Wicketkeeper-Batsman",
            "country": "India",
        },
    ),
    Document(
        page_content="Rohit Sharma is an explosive opening batsman and captain with multiple IPL titles.",
        metadata={
            "name": "Rohit Sharma",
            "team": "Mumbai Indians",
            "role": "Batsman",
            "country": "India",
        },
    ),
    Document(
        page_content="Jasprit Bumrah is a world-class fast bowler known for his yorkers and death-over bowling.",
        metadata={
            "name": "Jasprit Bumrah",
            "team": "Mumbai Indians",
            "role": "Bowler",
            "country": "India",
        },
    ),
    Document(
        page_content="Andre Russell is a powerful all-rounder known for explosive batting and fast bowling.",
        metadata={
            "name": "Andre Russell",
            "team": "Kolkata Knight Riders",
            "role": "All-rounder",
            "country": "West Indies",
        },
    ),
]




vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),

)

vector_store.add_documents(documents)
# data = vector_store.get(include=["embeddings", "documents", "metadatas"])
# print(data)
retriever=vector_store.as_retriever(search_kwargs={"k":2})
query="who is best Wicketkeeper-Batsman ?"


result=retriever.invoke(query)

for doc in result:
    print(doc.page_content)
    print('\n')



