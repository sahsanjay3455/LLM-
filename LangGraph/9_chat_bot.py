
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

model = ChatOpenAI(
    model="stepfun/step-3.5-flash",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
)

from  langgraph.graph.message import add_messages

class chatbot_state(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


def chat_model(state:chatbot_state):
    prompt=state['messages']

    response=model.invoke(prompt).content
    return {'messages':[response] }



#define checkpointer
conn=sqlite3.connect(database='chat_bot.db',check_same_thread=False)


checkpointer=SqliteSaver(conn)
#define graph
graph=StateGraph(chatbot_state)

#define node
graph.add_node('chat_model',chat_model)

#define edges
graph.add_edge(START,'chat_model')
graph.add_edge('chat_model',END)

#define compile

workflow=graph.compile(checkpointer=checkpointer)

