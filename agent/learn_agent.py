"""
地图搜索功能：1、获取地点的经纬度，2、根据经纬度能够获取到周边的店面信息。


实现：通过调用高德的api进行搜索。就是使用requests aiohttp方法进行调用
"""
from typing import Type

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from pydantic.v1 import BaseModel, Field
import requests
import aiohttp
from langchain_openai.chat_models import ChatOpenAI


from langchain.agents import create_openai_tools_agent,AgentExecutor


_ = load_dotenv("/Users/zhulang/work/llm/RAG/.env")


# 1、先编写agent需要使用的工具


class MapSearchInput(BaseModel):
    keyword: str = Field(..., description="搜索的地点")


class MapSearch(BaseTool):
    name = "MapSearch"
    description = "地图搜索功能：获取地点的经纬度"
    args_schema: Type[BaseModel] = MapSearchInput

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
        return res.text

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




llm = ChatOpenAI(model="gpt-4o",temperature=0.5)

prompt = ChatPromptTemplate.from_messages([
    ("system","你是一位地图通，你能够根据用户的指令去完成地图搜索功能，并且能够根据用户的指令去搜索周边的信息"),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

map_agent = create_openai_tools_agent(llm=llm, tools=[MapSearch(), searchAround()],prompt=prompt)
map_agent_executor = AgentExecutor(agent=map_agent, tools=[MapSearch(), searchAround()], verbose=True)



# map_agent_executor.invoke({"messages":[HumanMessage(content="请告诉我104.083766,30.630647周边有什么吃的？")]})
# map_agent_executor.ainvoke({"messages":[HumanMessage(content="请告诉我104.083766,30.630647周边有什么吃的？")]})

import asyncio

res = asyncio.run(map_agent_executor.ainvoke({"messages":[HumanMessage(content="请告诉我104.083766,30.630647周边有什么吃的？")]}))