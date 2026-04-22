
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create model
model1 = ChatOpenAI(
    model="stepfun/step-3.5-flash:free",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
)

model2 = ChatOpenAI(
    model="arcee-ai/trinity-large-preview:free",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="generate the report on the  topic:    {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="summarize the following text:{text}", input_variables=["text"]
)



report_generation_chain = RunnableSequence(prompt1, model2, parser)

branch_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 100,   # x is string now
        prompt2 | model1 | parser
    ),
    RunnablePassthrough()
)


final_chain = RunnableSequence(report_generation_chain, branch_chain)

result = final_chain.invoke({"topic": "AI"})

print(result)
