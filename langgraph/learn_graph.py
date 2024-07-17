"""
langgraph

点： 工具，model
线： 条件线，普通线   主要使用点（工具的指向）进行连接组成线
状态： 通过每次执行接收到用户的信息后，判断是走工具，还是END  去走对应的步骤
图：整个点线组成的 叫图。

"""

import os
from typing import Type
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langgraph.prebuilt import ToolNode

os.environ["OPENAI_API_BASE"] = "https://api.fe8.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-7ZOWMhdgnPW1flpLTAbY6yqUybh6mMz3wesyIeFtjliYTTp1"
class bingCarAcountInput(BaseModel):
    a: str = Field(..., description="账号名称")
    b: str = Field(..., description="车信息")


class bingCarAcount(BaseTool):
    args_schema: Type[BaseModel] = bingCarAcountInput
    description = "这是一个绑定账号和车信息的方法，需要用户提供账号名称和车信息，才能进行账号和车信息的绑定。如果用户没有提供账号名称或者车信息，则提示用户给出账号名称和车信息并再进行账号和车信息的绑定。不能单独使用用户提供的账号和车信息进行创建。只能走绑定"
    name = "bingCarAcount"

    async def _arun(self, a: str, b: str) -> str:
        print("异步调用了绑定账号和车信息的方法")
        return "绑定成功"

    def _run(self, a: str, b: str) -> str:
        print("调用了绑定账号和车信息的方法")
        return "绑定成功"


class createCarInput(BaseModel):
    a: str = Field(..., description="发动机")
    b: str = Field(..., description="底盘")
    c: str = Field(..., description="变速箱")


class createCar(BaseTool):
    args_schema: Type[BaseModel] = createCarInput
    description = "这是一个生成车信息的方法，需要用户提供发动机，底盘，变速箱信息才能进行造车。如果用户没有提供这些信息，或者缺少一些信息，则提示用户提供对应的信息直到需要的信息完整，才能进行造车。并把造车的信息返回给用户。"
    name = "createCar"

    async def _arun(self, a: str, b: str, c: str) -> str:
        print("异步调用了造成的方法")
        return "造车成功"

    def _run(self, a: str, b: str, c: str) -> str:
        print("调用了造车的方法")
        return "造车成功"



tools = [createCar(),bingCarAcount()]
tool_node = ToolNode(tools)

model = ChatOpenAI(temperature=0,model="gpt-4o",api_key="sk-fVxALjqWNXclhzatHRgPdAiNeRHezeSBb4PcqQ9RPBARMOjW",
                         base_url="https://api.fe8.cn/v1").bind_tools(tools)

def call_model(state: MessagesState) -> MessagesState:
    """
    这是调用大模型的方法
    :param state:
    :return:
    """
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages" : [response]}


def should_continue(state: MessagesState):
    """
    判断是否需要继续执行
    :param state:
    :return:
    """
    messages = state["messages"]
    if messages[-1].tool_calls:
        return "tools"
    else:
        return END


work_flow = StateGraph(MessagesState)

work_flow.add_node("tools",tool_node)
work_flow.add_node("model",call_model)


work_flow.add_edge("tools","model")

work_flow.add_conditional_edges("model",should_continue)

work_flow.set_entry_point("model")

app = work_flow.compile()

messages = [SystemMessage(content="你是一个造车助手，你擅造成，并能够把车和账号绑定"),
            HumanMessage(content="你好"),
            AIMessage(content="你好，请问有什么可以帮你")]


while True:
    human = input("请输入你的问题：")
    messages.append(HumanMessage(content=human))
    res = app.invoke({"messages":messages},config={"configurable":{"tread_id":1}})
    messages.append(AIMessage(content=res["messages"][-1].content))
    print(res["messages"][-1].content)




