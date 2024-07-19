import os
from typing import Type

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI


class createAccountInput(BaseModel):
    """create account tool"""
    a: str = Field(..., description="账号名称")


class createAccount(BaseTool):
    name = "createAccount"
    description = "这是一个生成账号的方法，用于给用户生成账号信息，如果用户没有提供账号的话，那么请自行生成一个账号名称，账号名称的规则是必须包含字母数字且不能少于8位，不能超过12位。生成账号后，请让用户确认生成的账号信息是否正确，如果用户确认了账号信息正确，则继续执行创建账号"
    args_schema: Type[BaseModel] = createAccountInput

    def _run(self, a: str) -> str:
        """Use the tool."""
        print("调用了这个生成账号的方法")
        return a

    async def _arun(self, a: str) -> str:
        """Use the tool."""
        print("异步调用了这个生成账号的方法")
        return a


class createOrderInput(BaseModel):
    """create account tool"""
    a: str = Field(..., description="账号名称")
    b: str = Field(..., description="工单创建时间")


class createOrder(BaseTool):
    name = "createOrder"
    description = "这是一个生成工单的方法，用于给用户生成工单信息，用户必须提供账号信息和工单创建时间，才能正常生成工单信息，并将创建成功的工单信息返回给用户。若用户没有提供账号或者工单创建时间，则提示用户给出账号和工单创建时间并再进行工单的创建"
    args_schema: Type[BaseModel] = createOrderInput

    def _run(self, a: str, b: str) -> str:
        """Use the tool."""
        print("调用了这个生成工单的方法")
        return a + "/" + b

    async def _arun(self, a: str, b: str) -> str:
        """Use the tool."""
        print("异步调用了这个生成工单的方法")
        return a + "/" + b


class bingAccountOrderInput(BaseModel):
    """bing account order tool"""
    a: str = Field(..., description="账号名称")
    b: str = Field(..., description="工单信息")


class bingAccountOrder(BaseTool):
    name = "bingAccountOrder"
    description = "这是一个绑定账号和工单的方法，用于给用户绑定账号和工单信息，用户必须提供账号信息和工单信息，才能正常绑定账号和工单信息，并将绑定成功的账号和工单信息返回给用户。若用户没有提供账号或者工单信息，则提示用户给出账号和工单信息并再进行账号和工单的绑定。不能单独使用用户提供的账号和工单信息进行创建。只能走绑定"
    args_schema: Type[BaseModel] = bingAccountOrderInput

    def _run(self, a: str, b: str) -> str:
        """Use the tool."""
        print("调用了这个绑定账号和工单的方法")
        return a + "/" + b

    async def _arun(self, a: str, b: str) -> str:
        """Use the tool."""
        print("异步调用了这个绑定账号和工单的方法")
        return a + "/" + b


os.environ["SERPAPI_API_KEY"] = "66425169b4775085d7ff617e6d547119482ee279211864ea8a78fd57e61a7d10"

os.environ["TAVILY_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "sk-nA4XFQzD7IZc8fVTcLDFqH1ds9ySyS39hpl46eOxiTltIfph"
os.environ["OPENAI_BASE_URL"] = "https://api.fe8.cn/v1"
tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()

create_account = createAccount()
create_order = createOrder()
bing_account_order = bingAccountOrder()


def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


members = ["Researcher", "Coder"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members
# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI(model="gpt-4-1106-preview")

supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
)

import functools
import operator
from typing import Sequence, TypedDict, Annotated

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import END, StateGraph, START


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


research_agent = create_agent(llm, [bingAccountOrder()],
                              "这是一个账号与订单绑定的助手，当用户需要把账号和订单绑定，请使用这个助手")
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
code_agent = create_agent(
    llm,
    [createAccount(), createOrder()],
    "这是一个创建订单和创建账号的助手，当用户需要创建账号，或者创建订单时，请使用这个助手",
)
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    workflow.add_edge(member, "supervisor")
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.add_edge(START, "supervisor")

graph = workflow.compile()

for s in graph.stream(
        {
            "messages": [
                HumanMessage(content="请帮我创建一个账号和生成一个订单，把订单和账号进行绑定")
            ]
        }
):
    print(s)
