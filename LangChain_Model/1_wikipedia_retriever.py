from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(k=2, lang='en')

query = "who is balen from nepal?"

docs = retriever.invoke(query)

for doc in docs:
    print(doc.page_content)
    print('\n')