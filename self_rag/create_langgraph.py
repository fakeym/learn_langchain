import time
from typing import List, TypedDict

from langgraph.constants import END
from langgraph.graph import StateGraph

from self_rag_tool import GradeAndGenerateTool


class CreateLangGraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]  # 检索后的信息，或者通过筛选后的信息


tools = GradeAndGenerateTool()


def retrieve(state):
    question = state["question"]
    documents = tools.search_vector(question)
    result_documents = []
    for info in documents[0]:
        result_documents.append(info["entity"]["text"])
    return {
        "documents": result_documents,
        "question": question,
    }


def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    result_documents = []
    for info in documents:
        result = tools.grade(question=question, text=info)
        if result == "yes":
            result_documents.append(info)
        else:
            continue
    return {"question": question, "documents": result_documents}


def generate_llm(state):
    question = state["question"]
    documents = state["documents"]
    documents_str = "\n".join(documents).replace("{", "").replace("}", "")
    result = tools.generate(question=question, text=documents_str)
    return {"question": question, "generation": result, "documents": documents}


def hallucinations_generate(state):
    print("开始调用幻觉判断方法")
    question = state["question"]
    generation = state["generation"]
    documents = state["documents"]
    documents_str = "\n".join(documents)
    result = tools.hallucinations(documents=documents_str, answer=generation)
    if result == "yes":
        return "generate_llm"
    else:
        print("判断问题和回复是否相关")
        generation = tools.answer_question(question=question, answer=generation)
        if generation == "yes":
            return "useful"
        else:
            return "rewrite_question"


def rewrite_question(state):
    question = state["question"]
    result = tools.rewrite_question(question=question)
    return {"question": result}


workflow = StateGraph(CreateLangGraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_llm", generate_llm)
workflow.add_node("rewrite_question", rewrite_question)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate_llm")
workflow.add_conditional_edges("generate_llm", hallucinations_generate,
                               {"generate_llm": "generate_llm", "rewrite_question": "rewrite_question", "useful": END})

workflow.add_edge("rewrite_question", "retrieve")

graph = workflow.compile()


start_time = time.time()
res = graph.stream({"question": "你们有什么菜品"})
for i in res:
    print(i)

print(time.time() - start_time)



state1 = {"question":"你们有什么菜品"}
state2 = {
        "documents": "向量检索结果",
        "question": "你们有什么菜品",
    }
state3 = {
        "documents": "只有与问题相关的向量检索结果",
        "question": "你们有什么菜品",
    }

state4 = {
        "documents": "只有与问题相关的向量检索结果",
        "question": "你们有什么菜品",
        "generation": "回复的结果",
    }
state5 = {
        "documents": "只有与问题相关的向量检索结果",
        "question": "重新复写的更加适合向量检索的问题",
        "generation": "回复的结果",
    }

