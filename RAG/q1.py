from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(
    model="stepfun/step-3.5-flash",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
)

while True:

    query=input("user:")
    if query=='exit':
        break

    print("AI",model.invoke(query).content)