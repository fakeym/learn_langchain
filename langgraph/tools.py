from typing import Type

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from rag_tool_graph import ChatDoc


chat = ChatDoc()

class queryInput(BaseModel):
    """query tool"""
    query: str = Field(..., description="问题")



class queryTool(BaseTool):
    name = "queryTool"
    description = "这是一个查询营地产品详情的工具，可以根据用户的问题，给出营地产品详情。如果查询结果不准确，可以使用其他工具进行回答"
    args_schema: Type[BaseModel] = queryInput

    def _run(self, query: str) -> str:
        print("调用query的方法进行回答")
        return chat.query_data(query)


    async def _arun(self, query: str) -> str:
        return chat.query_data(query)



class RagInput(BaseModel):
    question: str = Field(description="问题")


class RagTool(BaseTool):
    name = "RagTool"
    description = "这是一个RAG的工具，主要负责回答用户关于公司对于营地方面的产品问题和餐饮等问题"
    args_schema: Type[BaseModel] = RagInput

    def _run(self, question: str) -> str:
        print("调用RAG的方法进行回答")
        result = chat.ask_and_find(question)
        print(result)
        return result

    async def _arun(self, question: str) -> str:
        return chat.chat_with_doc(question)


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
