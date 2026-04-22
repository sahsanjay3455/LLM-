from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document




load_dotenv()

model = ChatOpenAI(
    model="stepfun/step-3.5-flash:free",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
)

prompt = PromptTemplate(
    template="you need to understand the given content:{content} and answer the question {question} from the content only and answer should be small and sweet",
    input_variables=["content",'question'],
)



documents = [

    Document(
        page_content="""The Grand Canyon is one of the most visited natural wonders in the world. Photosynthesis is the process by which green plants convert sunlight into energy. Millions of tourists travel to see it every year, and the rocks date back millions of years.""",
        metadata={"source": "Doc1"}
    ),

    Document(
        page_content="""In medieval Europe, castles were built primarily for defense. Knights wore armor made of metal, and siege weapons were often used to breach castle walls. The chlorophyll in plant cells captures sunlight during photosynthesis.""",
        metadata={"source": "Doc2"}
    ),

    Document(
        page_content="""Basketball was invented by Dr. James Naismith in the late 19th century. It was originally played with a soccer ball and peach baskets. The NBA is now a global league followed worldwide.""",
        metadata={"source": "Doc3"}
    ),

    Document(
        page_content="""The history of cinema began in the late 1800s with silent films. Thomas Edison was among the pioneers of early filmmaking. Modern filmmaking involves complex CGI and sound design, and photosynthesis does not occur in animal cells.""",
        metadata={"source": "Doc4"}
    )

]





embedding_model = OpenAIEmbeddings()

vector_store = Chroma(embedding_function=embedding_model)
vector_store.add_documents(documents)

retriever=vector_store.as_retriever(search_type="mmr",
search_kwargs={"k":2,"lambda_mult":0.25})
query="what is  Photosynthesis"
result=retriever.invoke(query)

for doc in result:
    print(doc.page_content)
    print("\n")







