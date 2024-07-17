import os
import json
from langchain.schema import (
    HumanMessage,
    FunctionMessage
)
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_BASE"] = "https://api.fe8.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-u8dqwXbP6SRnpYkna4HcCVpm6JAHEuLc28cXOuHSfI56PqGP"


def create_order():
    return "order"


def create_account():
    return "account"


def bing_order_account(order, account):
    return order + account

messages = []
def lang_chain_with_function_calling(text):
    functions = [{"name": "create_order", "description": "创建订单"},
                 {"name": "create_account", "description": "创建账户"},
                 {"name" : "bing_order_account","description" : "绑定账号和订单","parameters":{
                     "type": "object",
                     "properties": {
                         "order": {
                             "type": "string",
                             "description": "订单编号"
                         },
                         "account": {
                             "type": "string",
                             "description": "账户编号"
                         }
                     },
                     "required": ["order", "account"]
                 }}]

    messages.append(HumanMessage(content=text))
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    fun_map = {
        "create_order": create_order,
        "create_account": create_account,
        "bing_order_account": bing_order_account
    }
    message = llm.predict_messages(
        messages, functions=functions
    )

    status = message.response_metadata["finish_reason"]
    while status == "function_call":
        function_name = message.additional_kwargs["function_call"]["name"]
        arguments = json.loads(message.additional_kwargs["function_call"]["arguments"])
        function_response = fun_map[function_name](**arguments)

        function_message = FunctionMessage(name=function_name, content=function_response)
        messages.append(function_message)

        message = llm.predict_messages(
            messages=messages, functions=functions
        )
        status = message.response_metadata["finish_reason"]
    return "AI的回答: " + message.content + function_message



