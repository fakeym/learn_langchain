"""
举例：1、高德搜索附近的店  2、高德获取地点的经纬度。  3、RAG功能

"""
import functools
import operator
from typing import Type, TypedDict, Annotated, Sequence

import aiohttp
import requests
from dotenv import load_dotenv
from langchain_community.output_parsers.ernie_functions import JsonOutputFunctionsParser
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.tools import BaseTool
from learn_langgraph.constants import END
from learn_langgraph.graph import StateGraph,MessagesState
from pydantic.v1 import BaseModel, Field

from llm.RAG.rag_tool import ragTool
from langchain.agents import create_openai_tools_agent, AgentExecutor

_ = load_dotenv("/Users/zhulang/work/llm/RAG/.env")

rag_tool = ragTool()


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
        # prompt = "请帮我整理以下内容中的名称，地址和距离，并按照地址与名称对应输出，且告诉距离多少米，内容:{}".format(
        #     res.json())
        # result = llm.invoke(prompt)
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


class ragToolInput(BaseModel):
    question: str = Field(..., description="用户的问题")


class ragTool(BaseTool):
    args_schema: Type[BaseModel] = ragToolInput
    description = "这是一个RAG工具，需要用户提供问题，才能进行RAG的工具。如果用户没有提供问题，则提示用户给出问题并再进行RAG的工具。"
    name = "ragTool"

    def _run(self, question):
        return rag_tool.get_answer(question)


def create_agent(llm, tools, system_prompt):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt), MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["output"], name=name)]
    }


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
get_location_agent = create_agent(llm=llm, tools=[getLocation()],
                                  system_prompt="你是一个获取地点经纬度的助手，当用户需要获取经纬度时，你需要准确提供经纬度信息")
get_location_node = functools.partial(agent_node, agent=get_location_agent, name="get_location_agent")
search_around_agent = create_agent(llm=llm, tools=[searchAround()],
                                   system_prompt="你是一个地图通，你能够根据提供的经纬度去搜索周边店面信息。并返回给用户")
search_around_node = functools.partial(agent_node, agent=search_around_agent, name="search_around_agent")
rag_agent = create_agent(llm=llm, tools=[ragTool()],
                         system_prompt="你是一个RAG工具，主要是负责营地套餐的搜索")
rag_node = functools.partial(rag_agent, agent=rag_agent, name="rag_agent")

member = ["search_around_agent", "get_location_agent", "rag_agent"]

system_prompt = f"""
            你是一名任务管理者，负责管理任务的调度，下面是你的工作者{member},给定以下请求，与工作者一起响应，并采取下一步行动。
            每个工作者将执行一个任务并回复执行后的结果和状态，若全部执行完后，用FINISH回应。
            """

options = member + ["FINISH"]

function_def = {
    "name": "route",
    "description": "选择下一个工作者",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {
                        "enum": options
                    }
                ],
            }
        },
        "required": ["next"]
    }
}

prompt = ChatPromptTemplate.from_messages(
    [("system",system_prompt),MessagesPlaceholder(variable_name="messages"),
     ("system",f"基于上述的对话接下来应该是谁来采取行动？或者告诉我们应该完成吗？请在以下选项中进行选择{options}")]
).partial(options=str(options),member=",".join(member))


supervisor_chain = prompt | llm.bind_functions(functions=[function_def],function_call="route") | JsonOutputFunctionsParser()



class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],operator.add]
    next : str



work_flow = StateGraph(AgentState)

work_flow.add_node("get_location_agent",get_location_node)
work_flow.add_node("search_around_agent",search_around_node)
work_flow.add_node("rag_agent",rag_node)
work_flow.add_node("supervisor",supervisor_chain)

for name in member:
    work_flow.add_edge(name,"supervisor")

conditional_map = {
    "get_location_agent":"get_location_agent",
    "search_around_agent":"search_around_agent",
    "rag_agent":"rag_agent",
    "FINISH":END,
}

work_flow.add_conditional_edges("supervisor",lambda x : x["next"],conditional_map)

work_flow.set_entry_point("supervisor")

graph = work_flow.compile()


res = graph.invoke({"messages":[HumanMessage(content="请告诉我北京天安门附近有什么吃的")]})
print(res)



