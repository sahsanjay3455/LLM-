from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence ,RunnableParallel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create model
model1 = ChatOpenAI(
    model="stepfun/step-3.5-flash:free",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)

model2 = ChatOpenAI(
    model="arcee-ai/trinity-large-preview:free",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7
)

parser = StrOutputParser()

prompt1=PromptTemplate(
    template='generate the tweet about topic:{topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='generate the linked post about topic:{topic}',
    input_variables=['topic']
)


parallel_chain=RunnableParallel(
    {
        'tweet':RunnableSequence(prompt1,model1,parser),
        'linkdin':RunnableSequence(prompt2,model2,parser)
    }
)

print(parallel_chain.invoke({'topic':'AI'}))