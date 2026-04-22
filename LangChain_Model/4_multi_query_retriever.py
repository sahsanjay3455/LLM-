from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# from langchain.retrievers.multi_query import MultiQueryRetriever


load_dotenv()

model = ChatOpenAI(
    model="stepfun/step-3.5-flash",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
)




documents = [
    Document(
        page_content="Drinking enough water daily helps maintain hydration and supports overall body functions.",
        metadata={"category": "Hydration", "type": "Wellness Tip"},
    ),
    Document(
        page_content="Regular exercise improves cardiovascular health and boosts mental well-being.",
        metadata={"category": "Fitness", "type": "Health Tip"},
    ),
    Document(
        page_content="Eating a balanced diet rich in fruits and vegetables strengthens the immune system.",
        metadata={"category": "Nutrition", "type": "Diet Tip"},
    ),
    Document(
        page_content="Adequate sleep is essential for memory, mood, and overall physical health.",
        metadata={"category": "Sleep", "type": "Wellness Tip"},
    ),
    Document(
        page_content="Practicing meditation daily can reduce stress and improve mental clarity.",
        metadata={"category": "Mental Health", "type": "Mindfulness"},
    ),
    Document(
        page_content="Limiting sugar intake helps prevent obesity and reduces the risk of diabetes.",
        metadata={"category": "Nutrition", "type": "Health Advice"},
    ),
    Document(
        page_content="Maintaining good posture can prevent back pain and improve body alignment.",
        metadata={"category": "Posture", "type": "Physical Health"},
    ),
    Document(
        page_content="Regular health checkups help in early detection and prevention of diseases.",
        metadata={"category": "Preventive Care", "type": "Medical Advice"},
    ),
    Document(
        page_content="Spending time outdoors can improve mood and increase vitamin D levels.",
        metadata={"category": "Lifestyle", "type": "Wellness Tip"},
    ),
    Document(
        page_content="Deep breathing exercises can help calm the mind and reduce anxiety levels.",
        metadata={"category": "Mental Health", "type": "Relaxation"},
    ),
]

embedding_model = OpenAIEmbeddings()

vector_store = FAISS.from_documents(documents=documents, embedding=embedding_model)




#by using multi query retriever

query="what i need to do for healthy"



retriver=vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={"k":5,"lambda_mult":0.5},

)



result=retriver.invoke(query)



summarize_content=""

for doc in result:
    summarize_content+=doc.page_content



prompt = PromptTemplate(
    template="""you need to understand the given content:{content} and answer the question {question} from the content only and answer should be small and sweet""",
    input_variables=['content','question']
)


parser=StrOutputParser()

chain=prompt|model|parser

final_result=chain.invoke({'content':summarize_content,'question':query})

print('\n\n')

print(final_result)














# direct similarity_search
# result = vector_store.similarity_search("what is i need to do for healthy", k=2)
# for doc in result:
#     print(doc.page_content)


#by using mmr retriever

# retriver=vector_store.as_retriever(
#     search_type='mmr',
#     search_kwargs={"k":5,"lambda_mult":0.5},

# )

# query="what i need to do for healthy"

# result=retriver.invoke(query)
# for doc in result:
#     print(doc.page_content)




# retriver=vector_store.as_retriever(
#     search_type='mmr',
#     search_kwargs={"k":5,"lambda_mult":0.5},

# )