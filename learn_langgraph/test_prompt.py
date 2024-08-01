from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate

test = 111

system_template = PromptTemplate.from_file("prompt.txt")
print(system_template.template.format(kwargs="xxxxx",args="test11111"))
print(system_template)


# messages = [SystemMessage(content=f"xxxxx{teach_rag}"),HumanMessage(content=f"ddd{teach_rag}")]
# print(messages)