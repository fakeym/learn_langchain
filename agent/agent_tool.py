from langchain_community.utilities import SerpAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, tool, create_openai_tools_agent, AgentExecutor,tools
from langchain_community.agent_toolkits.load_tools import load_tools
import os
import warnings

from flask_api.project.config import OPENAI_BASE_URL

warnings.filterwarnings("ignore", category=Warning)

os.environ["OPENAI_API_BASE"] = OPENAI_BASE_URL
os.environ["OPENAI_API_KEY"] = "empty"
os.environ["SERPAPI_API_KEY"] = "66425169b4775085d7ff617e6d547119482ee279211864ea8a78fd57e61a7d10"
@tool("my_search_tool")
def search(query):
    """
    需要获取实时信息或者不知道的信息的时候，才可以使用这个工具
    根据给定的查询执行搜索并返回结果。

    使用SerpAPIWrapper类来封装搜索操作，该类负责与外部API交互，
    提供了一个简洁的方法来获取搜索结果。

    参数:
    - query: 字符串，表示要搜索的查询关键词。

    返回:
    - result: 字典，包含搜索结果的详细信息，具体结构取决于SerpAPI的返回。
    """
    # 初始化SerpAPI的包装类，这个类负责实际的搜索操作。
    serp = SerpAPIWrapper()
    # 使用给定的查询执行搜索，并返回结果。
    result = serp.run(query)
    # 返回搜索结果。
    return result

chatmodel = ChatOpenAI(
    model="qwen2-7b-agent-instruct",
    temperature=0,
    # streaming=True,
    # verbose=True,
    )

tools = [search]
prompt = ChatPromptTemplate.from_messages([("system","你是一个智能助手"),("human", "{input}"),MessagesPlaceholder(variable_name="agent_scratchpad")])
agent = create_openai_tools_agent(chatmodel,tools,prompt)
agent_executor = AgentExecutor(agent=agent,tools=tools)

res = agent_executor.invoke({"input":"今天星期几"})
print(res)



