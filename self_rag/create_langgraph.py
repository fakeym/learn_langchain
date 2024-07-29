from typing import List, TypedDict, Type, Any, Annotated

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent,AgentExecutor
from langgraph.constants import END
from langgraph.graph import StateGraph,MessagesState
from pydantic.v1 import BaseModel, Field
import concurrent.futures
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
    print(result_documents)
    return {"question": question, "documents": result_documents}


def generate_llm(state):
    question = state["question"]
    documents = state["documents"]
    documents_str = "\n".join(documents).replace("{", "").replace("}", "")
    result = tools.generate(question=question, text=documents_str)
    return {"question": question, "generation": result, "documents": documents}


def hallucinations_generate(state):
    print("调用幻觉判断的方法")
    question = state["question"]
    generation = state["generation"]
    documents = state["documents"]
    documents_str = "\n".join(documents)
    result = tools.hallucinations(documents=documents_str, answer=generation)
    if result == "yes":
        return "generate_llm"
    else:
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

class graphInput(BaseModel):
    question: str = Field(..., description="问题")

class runGraph(BaseTool):

    args_schema: Type[BaseModel] = graphInput
    description = "你是一个营地信息获取的工具，能够获取到关于营地房型，酒水，餐饮套餐，地址，四季活动，当地特产，联系方式等信息。并且通过用户的问题来进行对应的信息检索，得到最终的回复"
    name = "runGraph"


    def _run(self, question):
        result = tools.search_vector(question)
        documents_end = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(tools.grade, question, text["entity"]["text"], result[0].index(text)) for text in
                       result[0]}
            for future in concurrent.futures.as_completed(futures):
                if future.result()[0] == "yes":
                    documents_end.append(result[0][future.result()[1]]["entity"]["text"])
        end_text = "\n".join(documents_end).replace("}", "").replace("{", "")
        result = tools.generate(question, end_text)
        return result


    async def _arun(self, question):
        result = tools.search_vector(question)
        documents_end = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(tools.grade, question, text["entity"]["text"], result[0].index(text)) for text in
                       result[0]}
            for future in concurrent.futures.as_completed(futures):
                if future.result()[0] == "yes":
                    documents_end.append(result[0][future.result()[1]]["entity"]["text"])
        end_text = "\n".join(documents_end).replace("}", "").replace("{", "")
        result = tools.generate(question, end_text)
        return result


llm = ChatOpenAI(temperature=0, model="gpt-4o")
class llmInvokeInput(BaseModel):

    messages: list[AnyMessage] = Field(description="历史对话的内容")


class llmInvoke(BaseTool):
    args_schema: Type[BaseModel] = llmInvokeInput
    description = "这是一个可以基于历史对话的内容用来回答用户问题。"
    name = "llmInvoke"


    def _run(self, messages):
        res = llm.invoke(messages)
        return res.content






system_prompt = ChatPromptTemplate.from_messages([
    ("system","你是一位营地智能客服，当用户询问关于营地的问题时，你能够使用对应的runGraph工具进行回答。并且你可以根据用户的问题来判定是否需要调用runGraph工具。"),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
agent = create_openai_tools_agent(llm=llm, tools=[runGraph(), llmInvoke()],prompt=system_prompt)
agents = AgentExecutor(agent=agent, tools=[runGraph(),llmInvoke()])
messages = []
while True:
    question = input("请输入问题：")
    messages.append(HumanMessage(content=question))
    res = agents.invoke({"messages":messages})
    messages.append(AIMessage(content=res["output"]))
    print(res["output"])
