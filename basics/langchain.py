import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
model = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

model.invoke(messages)

#
# StrOutputParser()
#

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

result = model.invoke(messages)
parser.invoke(result)

#
# Prompt Templates
#

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

result = prompt_template.invoke({"language": "italian", "text": "hi"})
result.to_messages()

#
# Vector stores
#

from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)

vectorstore.similarity_search("cat")
# Async query:
await vectorstore.asimilarity_search("cat")
# Note that providers implement different scores; Chroma here
# returns a distance metric that should vary inversely with
# similarity.
vectorstore.similarity_search_with_score("cat")

# Return documents based on similarity to a embedded query:
embedding = OpenAIEmbeddings().embed_query("cat")
vectorstore.similarity_search_by_vector(embedding)

#
# RunnableLambda, Retriever
#

from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result

retriever.batch(["cat", "shark"])

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

retriever.batch(["cat", "shark"])

#
# RunnablePassthrough
#

# 사람이 인지하기 좋은 단어들로 변경해주는 기능을 수행할 수 있는 예제
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o")

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])
# message를 템플릿 형태로 만들어주면, 사용자가 주는 단어가 적절하게 들어감
# from_messages로 사람 메세지를 주면, 지금처럼 chain이 만들어지고, 이 chain을 통해서 GPT에서 프롬프트를 넘겨주면, GPT가 사람이 인지할 수 있는 장문의 문장으로 만들어줌

# message 문자열을 langchain의 템플릿 객체로 바꿔줌

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
response = rag_chain.invoke("tell me about cats")
print(f'response.content: {response.content}')

#
# Agent
#

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage

search = TavilySearchResults(max_results=2)
search_results = search.invoke("what is the weather in SF")
print(search_results)
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]

model = ChatOpenAI(model="gpt-4o")
# response = model.invoke([HumanMessage(content="hi!")])
# response.content

model_with_tools = model.bind_tools(tools)
# response = model_with_tools.invoke([HumanMessage(content="Hi!")])

# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}")

response = model_with_tools.invoke([HumanMessage(content="What's the weather in SF?")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")

#
# langgraph
#

from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(model, tools)
# response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})
# print(f'reponse["messages]: {response["messages"]}')

response = agent_executor.invoke({"messages": [HumanMessage(content="whats the weather in sf?")]})
print(f'reponse["messages]: {response["messages"]}')

#
# Streaming
#

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
    ):
    print(chunk)
    print("----")

async for event in agent_executor.astream_events(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}, version="v1"
    ):
    kind = event["event"]
    if kind == "on_chain_start":
        if (
            event["name"] == "Agent"
        ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
            print(
                f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
            )
    elif kind == "on_chain_end":
        if (
            event["name"] == "Agent"
        ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
            print()
            print("--")
            print(
                f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
            )
    if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            # Empty content in the context of OpenAI means
            # that the model is asking for a tool to be invoked.
            # So we only print non-empty content
            print(content, end="|")
    elif kind == "on_tool_start":
        print("--")
        print(
            f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
        )
    elif kind == "on_tool_end":
        print(f"Done tool: {event['name']}")
        print(f"Tool output was: {event['data'].get('output')}")
        print("--")

#
# Session management system
#

from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string(":memory:") as memory:
    agent_executor = create_react_agent(model, tools, checkpointer=memory)
    config = {"configurable": {"thread_id": "abc123"}}
    
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="hi im bob!")]}, config
    ):
        print(chunk)
        print("----")
        
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="whats my name?")]}, config
    ):
        print(chunk)
        print("----")

with SqliteSaver.from_conn_string(":memory:") as memory:
    agent_executor = create_react_agent(model, tools, checkpointer=memory)
    config = {"configurable": {"thread_id": "xyz123"}}
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="whats my name?")]}, config
    ):
        print(chunk)
        print("----")