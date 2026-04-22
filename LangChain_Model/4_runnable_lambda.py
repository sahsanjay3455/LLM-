# runnable_lambda help to work python function and combine with runnable to run the any pipeline

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

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
    template="generate the joke about topic:{topic}", input_variables=["topic"]
)


def word_counter(text):
    split_txt = text.split()
    cnt = 0
    for word in split_txt:
        cnt += 1
    return cnt


joke_chain = RunnableSequence(prompt1, model2, parser)
parallel_chain = RunnableParallel(
    {"joke": RunnablePassthrough(), "word_count": RunnableLambda(word_counter)}
)
final_chain = RunnableSequence(joke_chain, parallel_chain)

result = final_chain.invoke({"topic": "AI"})

print("""Joke:{} \n\n word count:{} """.format(result['joke'],result['word_count']))
