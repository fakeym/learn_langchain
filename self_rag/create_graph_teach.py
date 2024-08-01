from typing import TypedDict, List

from langgraph.constants import END
from langgraph.graph import StateGraph

# from self_rag.self_rag_tool_teach import gradeAndGenerateRagTool
import concurrent.futures

from llm.self_rag.self_rag_tool_teach import gradeAndGenerateRagTool

class CreateLanggraphState(TypedDict):
    question : str
    answer : str
    documents : List[str]




tools = gradeAndGenerateRagTool()



def retrieve(state):
    question = state["question"]
    print(state)
    documents = tools.search_vector(question)
    result_documents = []
    for document in documents[0]:
        result_documents.append(document["entity"]["text"])
    return {"question":question,"documents":result_documents}


def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    filtter_documents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(tools.grade,question,document,documents.index(document)) for document in documents}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result[0] == "yes":
                filtter_documents.append(documents[result[1]])

    return {"question":question,"documents":filtter_documents}


def generation(state):
    question = state["question"]
    documents = state["documents"]
    document = "\n".join(documents).replace("{","").replace("}","")
    answer = tools.generate(question,document)
    return {"question":question,"answer":answer}


def rewrite_question(state):
    question = state["question"]
    rewrite_question = tools.rewrite_question(question)
    return {"question":rewrite_question}



def grade_generation(state):
    documents = state["documents"]
    if documents:
        return "generation"
    return "rewrite_question"


def hallucinations_generate(state):
    documents = state["documents"]
    answer = state["answer"]
    question = state["question"]
    hallucinations_score = tools.hallucinations(documents,answer)
    if hallucinations_score == "yes":
        return "generation"
    else:
        answer_score = tools.answer_question(question,answer)
        if answer_score == "yes":
            return "useful"
        else:
            return "rewrite_question"



class createGraph(object):
    def create_graph(self):
            workflow = StateGraph(CreateLanggraphState)
            workflow.add_node("retrieve",retrieve)
            workflow.add_node("grade_documents",grade_documents)
            workflow.add_node("generation",generation)
            workflow.add_node("rewrite_question",rewrite_question)

            workflow.set_entry_point("retrieve")
            workflow.add_edge("retrieve","grade_documents")
            workflow.add_conditional_edges("grade_documents",grade_generation,{"generation":"generation","rewrite_question":"rewrite_question"})
            workflow.add_conditional_edges("generation",hallucinations_generate,{"generation":"generation","rewrite_question":"rewrite_question","useful":END})
            workflow.add_edge("rewrite_question","retrieve")

            graph = workflow.compile()
            return graph





