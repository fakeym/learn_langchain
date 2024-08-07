import asyncio
import functools
import json
import operator
import os
from typing import Type, TypedDict, Annotated, Sequence
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
import aiohttp
import requests
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langgraph.constants import END
from langgraph.graph import StateGraph
from pydantic.v1 import BaseModel, Field
from dotenv import load_dotenv


_ = load_dotenv("/Users/zhulang/work/llm/self_rag/.env")

llm_4o = ChatOpenAI(temperature=0, model="gpt-4o")
llm_3 = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")


class additionInput(BaseModel):
    x: int = Field(...)
    y: int = Field(...)


class addition(BaseTool):
    args_schema: Type[BaseModel] = additionInput
    description = "这是一个加法计算器，需要用户提供两个数字，才能进行加法计算。如果用户没有提供两个数字，则提示用户给出两个数字并再进行加法计算。"
    name = "addition"
    async def _arun(self, x,y):
        return x + y
    def _run(self, x,y):
        return x + y


class multiplicationInput(BaseModel):
    x: int = Field(..., description="第一个数字")
    y: int = Field(..., description="第二个数字")


class multiplication(BaseTool):
    args_schema: Type[BaseModel] = multiplicationInput
    description = "这是一个乘法计算器，需要用户提供两个数字，才能进行乘法计算。如果用户没有提供两个数字，则提示用户给出两个数字并再进行乘法计算。"
    name = "multiplication"

    async def _arun(self, x,y):
        return x * y

    def _run(self, x,y):
        return x * y



class subtractionInput(BaseModel):
    x: int = Field(..., description="第一个数字")
    y: int = Field(..., description="第二个数字")

class subtraction(BaseTool):
    args_schema: Type[BaseModel] = subtractionInput
    description = "这是一个减法计算器，需要用户提供两个数字，才能进行减法计算。如果用户没有提供两个数字，则提示用户给出两个数字并再进行减法计算。"
    name = "subtraction"

    async def _arun(self, x,y):
        return x - y

    def _run(self, x,y):
        return x - y








def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools,verbose=True)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["output"], name=name)],
    }


members = ["addition_multiplication", "subtraction"]

system_prompt = f"""
            你是一名任务管理者，负责管理任务的调度。下面都是你的工作者{members},给定以下请求：与工人一起响应以采取下一步行动，
            每个工人讲执行一个任务并回复执行后的结果和状态。若全部执行完后，用FINISH回应
"""
options = ["FINISH"] + members

function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", f"基于上述对话，接下来应该是谁来采取行动？或者告诉我们应该完成吗？请在以下选项中进行选择{options}")
    ]
).partial(options=str(options), members=",".join(members))

supervisor_chain = prompt | llm_4o.bind_functions(functions=[function_def], function_call="route") |JsonOutputFunctionsParser()




addition_multiplication_agent = create_agent(llm=llm_4o, tools=[addition(),multiplication()],
                                  system_prompt="你是一位数学家，主要负责计算加法和乘法相关的计算，精通数学的运算法则")
addition_multiplication_agent_node = functools.partial(agent_node, agent=addition_multiplication_agent, name="addition_multiplication")
subtraction_agent = create_agent(llm=llm_4o, tools=[subtraction()],
                                  system_prompt="你是一位数学家，主要负责计算减法相关的计算，精通数学的运算法则")
subtraction_agent_node = functools.partial(agent_node, agent=subtraction_agent, name="subtraction")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str




work_flow = StateGraph(AgentState)
work_flow.add_node("addition_multiplication", addition_multiplication_agent_node)
work_flow.add_node("subtraction", subtraction_agent_node)
work_flow.add_node("supervisor", supervisor_chain)
for function_name in members:
    work_flow.add_edge(function_name, "supervisor")

conditional_map = {
    "FINISH": END,
    "addition_multiplication": "addition_multiplication",
    "subtraction": "subtraction",
}






work_flow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

work_flow.set_entry_point("supervisor")

graph = work_flow.compile()



res = graph.stream({"messages": [HumanMessage(content="1+2*4-3等于几")]})
for i in res:
    print(i)



# res = graph.invoke({"messages": [HumanMessage(content="请告诉我四川大学望江校区的周边有什么餐饮店")]})
# print(res)

# res = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="请告诉我四川大学望江校区的周边有什么餐饮店")]}))

