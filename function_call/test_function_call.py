import json
import time

import requests
from langchain_community.llms import Ollama
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from flask_api.RAG.config import base_url

llm = Ollama(base_url=base_url, model="qwen2:72b", temperature=0.7)


# 通过高德API接口和大模型的能力，得到具体的店面距离等信息
def search_around(keyword, location):
    around_url = "https://restapi.amap.com/v5/place/around"
    params = {
        "key": "df8ff851968143fb413203f195fcd7d7",
        "keywords": keyword,
        "location": location
    }

    res = requests.get(url=around_url, params=params)
    prompt = "请帮我整理以下内容中的名称，地址和距离，并按照地址与名称对应输出，且告诉距离多少米，内容:{}".format(
        res.json())
    result = llm.invoke(prompt)
    return result + "\nend"


def get_location(keyword):
    url = "https://restapi.amap.com/v5/place/text"
    params = {
        "key": "df8ff851968143fb413203f195fcd7d7",
        "keywords": keyword,
    }
    res = requests.get(url=url, params=params)
    return '{}的经纬度是：'.format(keyword) + res.json()["pois"][0]["location"]


# 通过输入某某地址的某某行业的店。通过function_call的方式来得到最终结果


function_ollama = OllamaFunctions(base_url=base_url, model="qwen2:72b")

llm_tools = function_ollama.bind_tools(tools=[{
    "name": "get_location",
    "description": "根据用户输入的地理位置，使用高德的API接口查询出对应的经纬度",
    "parameters": {
        "keyword": {
            "type": "string",
            "description": "用户输入的地理位置",
        },
        "required": ["keyword"]
    }
},
    {"name": "search_around",
     "description": "根据提供的经纬度和行业类型，使用高德API接口搜索出附近对应的行业店铺",
     "parameters": {
         "keyword": {
             "type": "string",
             "description": "行业的关键字",
         },
         "location": {
             "type": "string",
             "description": "根据get_location方法提供的经纬度",
         }
     },
     "required": ["keyword", "location"]
     }])

function_map = {
    "search_around": search_around,
    "get_location": get_location
}

prompt = "请帮我找到北京大学附近的餐饮店"

messages = [
    {"role": "system", "content": "你是一位地图通，负责帮用户找到任何可以提供的地址"},
    {"role": "human", "content": prompt}
]
start_time = time.time()
res = llm_tools.invoke(json.dumps(messages,ensure_ascii=False))
def_names = json.loads(res.json())["tool_calls"]
result = "大模型未使用function_call"
while def_names:
    for def_name in def_names:
        name = def_name["name"]
        function_action = function_map.get(name, None)
        if function_action:
            result = function_action(**def_name['args'])
            print(result)
            if 'end' not in result:
                messages.append(
                    {"role": "tool",
                     "tool_call_id": def_name['id'],
                     "name": name,
                     "content": str(result)}
                )
                res = llm_tools.invoke(json.dumps(messages,ensure_ascii=False))
                def_names = json.loads(res.json())["tool_calls"]
            else:
                def_names = None

print(time.time() - start_time)