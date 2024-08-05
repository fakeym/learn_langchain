import concurrent.futures

from langchain_openai import ChatOpenAI

from llm.project.base_model import RouteQuery
from llm.project.self_rag_tool import GradeAndGenerateTool
from llm.project.vector_storage import VectorStorageObject

tools = GradeAndGenerateTool()
vector_tool = VectorStorageObject()


def get_knowledge_type(state):
    question = state["question"]
    filename = state["filename"]
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo").with_structured_output(RouteQuery)
    res = llm.invoke(question)
    return {"question": question, "collection_name": res.route,"filename":filename}


def retrieve(state):
    question = state["question"]
    collection_name = state["collection_name"]
    filename = state["filename"]
    documents = tools.search_vector(question, collection_name)
    result_documents = []
    for document in documents[0]:
        result_documents.append(document["entity"]["text"])
    return {"question": question, "documents": result_documents,"collection_name":collection_name,"filename":filename}


def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    filename = state["filename"]
    filtter_documents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(tools.grade, question, document, documents.index(document)) for document in
                   documents}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result[0] == "yes":
                filtter_documents.append(documents[result[1]])

    return {"question": question, "documents": filtter_documents,"filename":filename}


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
    return {"question": question, "documents": documents}


def file_out(state):
    return {"answer": "你提供的文档不支持回答"}


def route_node(state):
    filename = state["filename"]
    if filename:
        return "vector_storage"
    else:
        return "get_knowledge_type"
