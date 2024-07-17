import time

from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, Tool

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.tools import tool
from pydantic.v1 import BaseModel, Field

from flask_api.RAG.config import base_url, OPENAI_BASE_URL

import os

import warnings

warnings.filterwarnings("ignore", category=Warning)

os.environ["SERPAPI_API_KEY"] = "66425169b4775085d7ff617e6d547119482ee279211864ea8a78fd57e61a7d10"
# os.environ["OPENAI_API_KEY"] = "empty"
# os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL


def sum(string):
    x = string.split(",")[0]
    y = string.split(",")[1]
    return x + y


sum_fun = Tool.from_function(func=sum, name="sum",
                             description="这是一个只会算加法的计算器。当用户需要得到两个数的和时，请调用sum方法进行加法运算，且两个参数之间用逗号隔开，并返回运算后的结果")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key="sk-u8dqwXbP6SRnpYkna4HcCVpm6JAHEuLc28cXOuHSfI56PqGP",
    base_url="https://api.fe8.cn/v1"
)

tools = load_tools(["serpapi", "llm-math"], llm=llm)
tools.append(sum_fun)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,

)

start_time = time.time()
res = agent.stream({"input": "成都今天的天气怎么样"})
for i in res:
    print(i, end="", flush=True)
print(time.time() - start_time)
