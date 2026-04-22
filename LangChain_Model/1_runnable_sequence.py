from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create model
model = ChatOpenAI(
    model="stepfun/step-3.5-flash:free",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="what is joke on the topic:{topic} ", input_variables=["topic"]
)

prompt2=PromptTemplate(
    template='explain about the joke:{joke}',
    input_variables=['joke']
)

chain = RunnableSequence(prompt1, model, parser,prompt2,model,parser)
print(chain.invoke({"topic": "AI"}))
