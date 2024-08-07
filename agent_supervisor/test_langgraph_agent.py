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

class searchAroundInput(BaseModel):
    keyword: str = Field(..., description="搜索关键词")
    location: str = Field(..., description="地点的经纬度")


class searchAround(BaseTool):
    args_schema: Type[BaseModel] = searchAroundInput
    description = "这是一个搜索周边信息的方法，需要用户提供关键词和地点的经纬度，才能进行周边信息的搜索。如果用户没有提供关键词或者地点的经纬度，则提示用户给出关键词和地点的经纬度并再进行周边信息的搜索。"
    name = "searchAround"

    def _run(self, keyword, location):
        around_url = "https://restapi.amap.com/v5/place/around"
        params = {
            "key": "df8ff851968143fb413203f195fcd7d7",
            "keywords": keyword,
            "location": location
        }
        print("同步调用获取地点周边的方法")
        res = requests.get(url=around_url, params=params)
        prompt = "请帮我整理以下内容中的名称，地址和距离，并按照地址与名称对应输出，且告诉距离多少米，内容:{}".format(
            res.json())
        result = llm_3.invoke(prompt)
        return result.content

    async def _arun(self, keyword, location):
        async with aiohttp.ClientSession() as session:
            around_url = "https://restapi.amap.com/v5/place/around"
            params = {
                "key": "df8ff851968143fb413203f195fcd7d7",
                "keywords": keyword,
                "location": location
            }
            print("异步调用获取地点周边的方法")
            async with session.get(url=around_url, params=params) as response:
                return await response.json()


class getLocationInput(BaseModel):
    keyword: str = Field(..., description="搜索关键词")


class getLocation(BaseTool):
    args_schema: Type[BaseModel] = getLocationInput
    description = "这是一个获取地点的经纬度的方法，需要用户提供关键词，才能进行地点的经纬度的获取。如果用户没有提供关键词，则提示用户给出关键词并再进行地点的经纬度的获取。"
    name = "getLocation"

    def _run(self, keyword):
        url = "https://restapi.amap.com/v5/place/text"
        params = {
            "key": "df8ff851968143fb413203f195fcd7d7",
            "keywords": keyword,
        }
        res = requests.get(url=url, params=params)
        print("同步调用获取地点的经纬度方法")
        return '{}的经纬度是：'.format(keyword) + res.json()["pois"][0]["location"]

    async def _arun(self, keyword):
        async with aiohttp.ClientSession() as session:
            url = "https://restapi.amap.com/v5/place/text"
            params = {
                "key": "df8ff851968143fb413203f195fcd7d7",
                "keywords": keyword,
            }
            print("异步调用获取地点的经纬度方法")
            async with session.get(url=url, params=params) as response:
                res = await response.json()
                return '{}的经纬度是：'.format(keyword) + res["pois"][0]["location"]




class calculatorInput(BaseModel):
    x : str = Field(..., description="第一个数字")
    y : str = Field(..., description="第二个数字")


class calculator(BaseTool):
    args_schema: Type[BaseModel] = calculatorInput
    description = "你是一个高德的输入提示的智能工具，通过用户的关键词提示用户相关的地点"
    name = "calculator"


    def _run(self,keyword):
        around_url = "https://restapi.amap.com/v3/assistant/inputtips"
        params = {
            "key": "df8ff851968143fb413203f195fcd7d7",
            "keywords": keyword,
        }
        print("调用输入提示")
        res = requests.get(url=around_url, params=params)
        return res





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


members = ["search_around", "calculator"]

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




search_round_agent = create_agent(llm=llm_4o, tools=[searchAround(),getLocation()],
                                  system_prompt="你是一位地图通，你能够根据用户给出地点去获取到地点周边的地点信息，并返回给用户。你不会为用户提供加法计算的能力")
search_around_node = functools.partial(agent_node, agent=search_round_agent, name="search_around")
get_location_agent = create_agent(llm=llm_3, tools=[calculator()],
                                  system_prompt="你是一个高德的输入提示的智能工具，通过用户的关键词提示用户相关的地点")
calculator = functools.partial(agent_node, agent=get_location_agent, name="calculator")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str




work_flow = StateGraph(AgentState)
work_flow.add_node("calculator", calculator)
work_flow.add_node("search_around", search_around_node)
work_flow.add_node("supervisor", supervisor_chain)
for function_name in members:
    work_flow.add_edge(function_name, "supervisor")

conditional_map = {
    "FINISH": END,
    "search_around": "search_around",
    "calculator": "calculator",
}






work_flow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

work_flow.set_entry_point("supervisor")

graph = work_flow.compile()



res = graph.invoke({"messages": [HumanMessage(content="请告诉我成都环球中心的周边有什么餐饮店和请告诉我仙林相关的地点")]})
print(res)



# res = graph.invoke({"messages": [HumanMessage(content="请告诉我四川大学望江校区的周边有什么餐饮店")]})
# print(res)

# res = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="请告诉我四川大学望江校区的周边有什么餐饮店")]}))

