from langchain.chains.sequential import SequentialChain
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

from flask_api.RAG.config import base_url
from langchain.chains.llm import LLMChain

import warnings

warnings.filterwarnings("ignore", category=Warning)

llm = Ollama(base_url=base_url, model="qwen2:72b", temperature=0.6)
# res = llm.invoke("你是谁？")
# print(res)

prompt_template_1 = "请把以下内容翻译成中文，内容：{content}"
prompt = ChatPromptTemplate.from_template(prompt_template_1)
llm_chain_1 = LLMChain(llm=llm, prompt=prompt, verbose=True, output_key="chinese_content")

prompt_template_2 = "请对翻译后的内容进行总结或摘要，内容：{chinese_content}"
prompt = ChatPromptTemplate.from_template(prompt_template_2)
llm_chain_2 = LLMChain(llm=llm, prompt=prompt, verbose=True, output_key="chinese_summary")

prompt_template_3 = "以下内容是什么语言，内容：{chinese_summary}"
prompt = ChatPromptTemplate.from_template(prompt_template_3)
llm_chain_3 = LLMChain(llm=llm, prompt=prompt, verbose=True, output_key="language")

prompt_template_4 = "请使用指定的语言对以下内容进行评论，内容:{chinese_summary},语言:{language}"
prompt = ChatPromptTemplate.from_template(prompt_template_4)
llm_chain_4 = LLMChain(llm=llm, prompt=prompt, verbose=True, output_key="review")

prompt_template_5 = "请对把以下内容翻译成英文，并中英文双版本输出。内容：{review}"
prompt = ChatPromptTemplate.from_template(prompt_template_5)
llm_chain_5 = LLMChain(llm=llm, prompt=prompt, verbose=True, output_key="two_language")

sequent_chain = SequentialChain(chains=[llm_chain_1, llm_chain_2, llm_chain_3, llm_chain_4, llm_chain_5],
                                input_variables=["content"],
                                output_variables=["chinese_content", "chinese_summary", "language", "review","two_language"],
                                verbose=True)


content_en = """
Autumn is a season of profound beauty and transformation. As the warm summer days fade into cooler, crisp afternoons, the landscape is painted in a palette of rich, earthy colors. The leaves of the trees, once a vibrant green, now turn to shades of gold, orange, and red, creating a fiery display that lights up the sky.
The air is filled with the scent of falling leaves, a mix of sweet decay and fresh earth, a reminder of the cycle of life and death. The light becomes softer, casting golden hues across the land, giving everything a warm, cozy feel.
The days become shorter, and the nights longer, as the moon and stars seem to shine brighter in the clear autumn sky. The air is still, allowing the sounds of the season to carry far—the rustling of leaves, the chirping of birds, and the gentle rustle of the wind through the trees.
Autumn is a time of harvest, when the fruits of summer's labor are gathered and celebrated. Fields are filled with the bounty of the season—golden corn, red apples, and purple grapes, all ready to be enjoyed.
It is also a time for introspection and reflection, as the year winds down and we prepare for the colder months ahead. Autumn encourages us to take stock of our lives, to appreciate the beauty and bounty that surrounds us, and to cherish the memories we have made.
In short, autumn is a season of wonder and amazement, a time to appreciate the natural world in all its glory and to relish the simple pleasures of the changing season.
"""

res = sequent_chain(content_en)
print(res)
# """
# 1号链：有输入，有输出
# 2号链：有输入，有输出。2号链的输入，是从1号链的输出这样获取来的。
# 3号链：有输入，有输出。3号链的输入，是从2号链的输出这样获取来的。
# 4号链：有输入，有输出。4号链的输入，是从3号链的输出这样获取来的。
# 5号链：有输入，有输出。5号链的输入，是从4号链的输出这样获取来的。
# """



# 1 -2 -3 -4
