import time
from operator import add
from typing import TypedDict, List, Dict, Annotated

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph

from rag_tool_graph import ChatDoc

# _ = load_dotenv()


"""
下面是一个具有rag功能的子图
"""


class ragState(TypedDict):
    rag_result: str
    result: List[Dict]  # 这是父图获取用户的指令并传给子图，子图拿到后去执行对应的方法


def rag_tool(state):
    messages = state["result"][-1].content
    rag_result = ChatDoc().ask_and_find(messages)
    return {"rag_result": rag_result}


work_flow_rag = StateGraph(ragState)
work_flow_rag.add_node("rag_tool", rag_tool)
work_flow_rag.set_entry_point("rag_tool")
work_flow_rag.set_finish_point("rag_tool")

"""
下面是一个具有query功能的子图
"""


class queryState(TypedDict):
    query_result: str
    result: List[Dict]  # 这是父图获取用户的指令并传给子图，子图拿到后去执行对应的方法


def query_tool(state):
    messages = state["result"][-1].content
    query_result = ChatDoc().query_data(messages)
    return {"query_result": query_result}


work_flow_query = StateGraph(queryState)
work_flow_query.add_node("query_tool", query_tool)
work_flow_query.set_entry_point("query_tool")
work_flow_query.set_finish_point("query_tool")

"""
下面是一个父图
"""


class graphState(TypedDict):
    human_input: List[Dict]
    result: Annotated[List[Dict], add]
    rag_result: str
    query_result: str
    result_end: str


class runGraph(object):

    def __init__(self):
        self.system_template = PromptTemplate.from_file("prompt.txt")
        self.messages = [SystemMessage(content=self.system_template.template)]
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o",
                              api_key="sk-nA4XFQzD7IZc8fVTcLDFqH1ds9ySyS39hpl46eOxiTltIfph",
                              base_url="https://api.fe8.cn/v1")
        self.human_prompt = PromptTemplate.from_file("human_prompt.txt")
        print(self.human_prompt)

    def call_model(self, state):
        messages = state["human_input"]
        return {"result": messages}

    def end_model(self, state):
        human_content = self.human_prompt.template.format(question=state["human_input"][-1].content, rag_info=state["rag_result"], query_info=state["query_result"])
        self.messages.append(HumanMessage(content=human_content))
        res = self.llm.invoke(self.messages)
        self.messages.append(AIMessage(content=res.content))
        return {"result_end": res.content}

    def create_graph(self):
        work_flow = StateGraph(graphState)

        work_flow.add_node("call_model", self.call_model)
        work_flow.add_node("end_model", self.end_model)
        work_flow.add_node("query_graph", work_flow_query.compile())
        work_flow.add_node("rag_graph", work_flow_rag.compile())
        work_flow.set_entry_point("call_model")
        work_flow.add_edge("call_model", "query_graph")
        work_flow.add_edge("call_model", "rag_graph")
        work_flow.add_edge("query_graph", "end_model")
        work_flow.add_edge("rag_graph", "end_model")
        work_flow.add_edge("end_model", END)
        graph = work_flow.compile()
        return graph

    def run_langgraph(self, question):
        self.graph = self.create_graph()
        messages = [HumanMessage(content=question)]
        res = self.graph.invoke({"human_input": messages})
        return res["result_end"]


if __name__ == '__main__':
    graph = runGraph()
    while True:
        question = input("请输入问题：")
        start_time = time.time()
        res = graph.run_langgraph(question)
        print(res)
        print(time.time() - start_time)