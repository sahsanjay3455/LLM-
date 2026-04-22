from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence ,RunnableParallel,RunnablePassthrough
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
    template='generate the joke about topic:{topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='generate the explaination  about topic:{topic}',
    input_variables=['topic']
)


joke_chain=RunnableSequence(prompt1,model2,parser)
parallel_chain=RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'explaination':RunnableSequence(prompt2,model2,parser)
    }
)
final_chain=RunnableSequence(joke_chain,parallel_chain)

print(final_chain.invoke({'topic':'AI'}))


