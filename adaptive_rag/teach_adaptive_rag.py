import time
from typing import Literal, TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic.v1 import BaseModel, Field

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

from llm.self_rag.create_graph_teach import createGraph

graph_tool = createGraph()
create_graph_node = graph_tool.create_graph()

_ = load_dotenv("/Users/zhulang/work/llm/self_rag/.env")


class RouteQuery(BaseModel):
    """
    将用户查询路由到最相关的数据源
    """

    route: Literal["rag_node","now_time","calculator_node"] = Field(...,description="用户给定一个问题，选择路由到哪个数据源")


llm = ChatOpenAI(temperature=0,model="gpt-4o").with_structured_output(RouteQuery)

class graphState(TypedDict):

    question : str
    generation : str

def rag_node(state):
    print("调用了RAG")
    question = state["question"]
    result = create_graph_node({"question":question})
    return {"question":question,"generation":result}


def calculator_node(state):
    print("调用了计算器")
    question = state["question"]
    calculator = ChatOpenAI(temperature=0, model="gpt-4o")
    res = calculator.invoke(question)
    return {"question":question,"generation":res.content}

def now_time_node(state):
    print("调用了获取时间方法")
    question = state["question"]
    now_time_str = time.time()
    return {"question":question,"generation":now_time_str}


def route_node(state):
    question = state["question"]
    res = llm.invoke(question)
    return res.route





workflow = StateGraph(graphState)

workflow.add_node("rag_node",rag_node)
workflow.add_node("now_time",now_time_node)
workflow.add_node("calculator_node",calculator_node)

workflow.add_conditional_edges(START,route_node,{"rag_node":"rag_node","now_time":"now_time","calculator_node":"calculator_node"})
workflow.add_edge("rag_node",END)
workflow.add_edge("now_time",END)
workflow.add_edge("calculator_node",END)

graph_adaptive = workflow.compile()



if __name__ == '__main__':
    question = "请告诉我现在是几点"
    res = graph_adaptive.invoke({"question":question})
    print(res)