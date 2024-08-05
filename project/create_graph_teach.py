import time
from typing import TypedDict, List

from dotenv import load_dotenv
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from llm.project.base_node import get_knowledge_type, vector_storage, retrieve, file_out, grade_documents, generation, \
    rewrite_question, route_node, grade_generation, hallucinations_generate

_ = load_dotenv("/Users/zhulang/work/llm/self_rag/.env")


class CreateLanggraphState(TypedDict):
    question: str
    answer: str
    documents: List[str]
    collection_name: str
    filename: str


class createGraph(object):
    def create_graph(self):
        workflow = StateGraph(CreateLanggraphState)

        workflow.add_node("get_knowledge_type", get_knowledge_type)
        workflow.add_node("vector_storage", vector_storage)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("file_out", file_out)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generation", generation)
        workflow.add_node("rewrite_question", rewrite_question)

        workflow.add_conditional_edges(START, route_node,
                                       {"vector_storage": "vector_storage", "get_knowledge_type": "get_knowledge_type"})
        workflow.add_edge("vector_storage", "grade_documents")

        workflow.add_edge("get_knowledge_type", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges("grade_documents", grade_generation,
                                       {"generation": "generation", "rewrite_question": "rewrite_question",
                                        "file_out": "file_out"})
        workflow.add_conditional_edges("generation", hallucinations_generate,
                                       {"generation": "generation", "rewrite_question": "rewrite_question",
                                        "useful": END})
        workflow.add_edge("file_out", END)

        workflow.add_edge("rewrite_question", "retrieve")

        graph = workflow.compile()
        return graph



