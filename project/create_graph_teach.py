import concurrent.futures
from typing import TypedDict, List

from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from llm.project.base_model import RouteQuery
from llm.project.self_rag_tool import GradeAndGenerateTool
from dotenv import load_dotenv

from llm.project.vector_storage import VectorStorageObject

_ = load_dotenv("/Users/zhulang/work/llm/self_rag/.env")


vector_tool = VectorStorageObject()

class CreateLanggraphState(TypedDict):
    question: str
    answer: str
    documents: List[str]
    collection_name: str
    filename : str



tools = GradeAndGenerateTool()



def get_knowledge_type(state):
    question = state["question"]
    llm = ChatOpenAI(temperature=0, model="gpt-4o").with_structured_output(RouteQuery)
    res = llm.invoke(question)
    print(res.route)
    return {"question":question,"collection_name":res.route}



def retrieve(state):
    question = state["question"]
    collection_name = state["collection_name"]
    documents = tools.search_vector(question,collection_name)
    result_documents = []
    for document in documents[0]:
        result_documents.append(document["entity"]["text"])
    return {"question": question, "documents": result_documents}


def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    filtter_documents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(tools.grade, question, document, documents.index(document)) for document in
                   documents}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result[0] == "yes":
                filtter_documents.append(documents[result[1]])

    return {"question": question, "documents": filtter_documents}


def generation(state):
    question = state["question"]
    documents = state["documents"]
    document = "\n".join(documents).replace("{", "").replace("}", "")
    answer = tools.generate(question, document)
    return {"question": question, "answer": answer}


def rewrite_question(state):
    question = state["question"]
    rewrite_question = tools.rewrite_question(question)
    return {"question": rewrite_question}


def grade_generation(state):
    documents = state["documents"]
    filename = state["filename"]
    if documents:
        return "generation"
    else:
        if filename:
            return "file_out"
        return "rewrite_question"


def hallucinations_generate(state):
    documents = state["documents"]
    answer = state["answer"]
    question = state["question"]
    hallucinations_score = tools.hallucinations(documents, answer)
    if hallucinations_score == "yes":
        return "generation"
    else:
        answer_score = tools.answer_question(question, answer)
        if answer_score == "yes":
            return "useful"
        else:
            return "rewrite_question"

def vector_storage(state):
    question = state["question"]
    filename = state["filename"]
    documents = vector_tool.split_text(filename)
    return {"question":question,"documents":documents}


def file_out(state):
    return {"answer":"你提供的文档不支持回答"}


def route_node(state):
    filename = state["filename"]
    if filename:
        return "vector_storage"
    else:
        return "get_knowledge_type"


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

        workflow.add_conditional_edges(START, route_node,{"vector_storage":"vector_storage","get_knowledge_type":"get_knowledge_type"})
        workflow.add_edge("vector_storage","grade_documents")

        workflow.add_edge("get_knowledge_type","retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges("grade_documents", grade_generation,
                                       {"generation": "generation", "rewrite_question": "rewrite_question","file_out":"file_out"})
        workflow.add_conditional_edges("generation", hallucinations_generate,
                                       {"generation": "generation", "rewrite_question": "rewrite_question",
                                        "useful": END})
        workflow.add_edge("file_out", END)

        workflow.add_edge("rewrite_question", "retrieve")

        graph = workflow.compile()
        return graph




if __name__ == '__main__':
    graph = createGraph().create_graph()
    res = graph.stream({"question":"请告诉我电冰箱的发展史","filename":"电视机.docx"})
    for i in res:
        print(i)



