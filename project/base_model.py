from typing import Literal

from dotenv import load_dotenv
from pydantic.v1 import BaseModel, Field

_ = load_dotenv("/Users/zhulang/work/llm/self_rag/.env")


class RouteQuery(BaseModel):
    """
    将用户查询路由到最相关的数据源
    """

    route: Literal["refrigerator", "air_conditioning", "TV"] = Field(...,
                                                                     description="用户给定一个问题，选择路由到哪个数据源")


class GradedRagTool(BaseModel):
    """
    对检索到到文档进行相关性的检查，相关返回yes，不相关返回no
    """

    binary_score: str = Field(description="文档与问题的相关性，'yes' or 'no'")


class GradeHallucinations(BaseModel):
    """
    对最终对回答进行一个判断，判断回答中是否存在幻觉，存在则输出yes，不存在这输出no
    """

    binary_score: str = Field(description="问题与回答的相关性，'yes' or 'no'")


class GradeAnswer(BaseModel):
    """对最终的回答于问题进行比对，判断回答和问题是相关的，是相关的则输出yes，不相关则输出no"""

    binary_score: str = Field(
        description="问题与回答的相关性， 'yes' or 'no'"
    )
