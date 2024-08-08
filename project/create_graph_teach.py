import time
from typing import TypedDict, List, Type

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic.v1 import BaseModel, Field

from llm.project.base_node import get_knowledge_type, vector_storage, retrieve, file_out, grade_documents, generation, \
    rewrite_question, route_node, grade_generation, hallucinations_generate,end_answer
from langchain.agents import create_openai_tools_agent,AgentExecutor
from llm.project.config import base_url

_ = load_dotenv("/Users/zhulang/work/llm/self_rag/.env")


class CreateLanggraphState(TypedDict):
    question: str
    answer: str
    documents: List[str]
    collection_name: str
    filename: str
    hallucination_count : int

class CreateLanggraphInput(BaseModel):
    question: str = Field(..., description="问题")
    filename : str = Field(None, description="文件名")



class createGraph(BaseTool):
    args_schema: Type[BaseModel] = CreateLanggraphInput
    description = "这是一个有关于家电领域的智能回答工具，只支持回答电视机，冰箱，空调的问答，请根据用户的问题给出回答，如果用户给出的问题不属于上述三个领域，则给出提示，并重新提问。"
    name = "create_graph"
    def _run(self,question,filename):
        workflow = StateGraph(CreateLanggraphState)

        workflow.add_node("get_knowledge_type", get_knowledge_type)
        workflow.add_node("vector_storage", vector_storage)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("file_out", file_out)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generation", generation)
        workflow.add_node("rewrite_question", rewrite_question)
        workflow.add_node("end_answer", end_answer)

        workflow.add_conditional_edges(START, route_node,
                                       {"vector_storage": "vector_storage", "get_knowledge_type": "get_knowledge_type"})
        workflow.add_edge("vector_storage", "grade_documents")

        workflow.add_edge("get_knowledge_type", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges("grade_documents", grade_generation,
                                       {"generation": "generation", "rewrite_question": "rewrite_question",
                                        "file_out": "file_out"})
        workflow.add_conditional_edges("generation", hallucinations_generate,
                                       {"generation": "generation", "rewrite_question": "rewrite_question","end_answer":"end_answer",
                                        "useful": END})
        workflow.add_edge("file_out", END)

        workflow.add_edge("rewrite_question", "retrieve")

        graph = workflow.compile()
        result = graph.invoke({"question": question, "filename": filename})

        return result["answer"]






class CreateLLMCustomerService(object):


    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="qwen2-instruct",base_url=base_url, api_key="xxx")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位关于家电行业的智能客服机器人，你可以回答冰箱，空调，电视的问题，请根据用户的问题给出回答，如果用户给出的问题不属于上述三个领域，则给出提示，并重新提问。"),
            # ("system", "你是一位数学家，你会计算两个数的和与两个数的乘积，并且能把结果返回个给用户"),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        self.agent = create_openai_tools_agent(llm=self.llm,tools=[createGraph()],prompt=prompt)
        self.excutor_agent = AgentExecutor(agent=self.agent, tools=[createGraph()])
        self.messages = []


    def chat(self, question, filename):
        self.messages.append(HumanMessage(content=f"问题：{question},文件：{filename}"))
        res = self.excutor_agent.invoke({"messages": self.messages})
        self.messages.append(AIMessage(content=res["output"]))
        return res["output"]









