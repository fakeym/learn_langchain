from typing import Literal

from pydantic.v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv


_ = load_dotenv("/Users/zhulang/work/llm/self_rag/.env")


class RouteQuery(BaseModel):
    """
    将用户查询路由到最相关的数据源
    """

    route: Literal["refrigerator","air_conditioning","TV"] = Field(...,description="用户给定一个问题，选择路由到哪个数据源")





