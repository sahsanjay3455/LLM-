
# env\Scripts\activate
# python 9_backend_chat_bot.py


from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

load_dotenv()

model = ChatOpenAI(
    model="stepfun/step-3.5-flash",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
)

# ---------------- STATE ---------------- #

class chatbot_state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ---------------- NODE ---------------- #

def chat_node(state: chatbot_state):
    messages = state["messages"]
    
    response = model.invoke(messages)
    
    return {
        "messages": [AIMessage(content=response.content)]
    }

# ---------------- GRAPH ---------------- #

checkpointer = MemorySaver()

graph = StateGraph(chatbot_state)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)

# ---------------- EXECUTION ---------------- #

# while True:
#     print('\n-------------\n')
#     user_input = input("user: ")
#     if user_input in['exit','bye','stop','okay']:
#         break

#     config = {"configurable": {"thread_id": "1"}}

#     initial_state = {
#         "messages": [HumanMessage(content=user_input)]
#     }

#     final_state = workflow.invoke(initial_state, config=config)

#     print("AI:", final_state["messages"][-1].content)