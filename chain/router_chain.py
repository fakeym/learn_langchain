from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router import MultiPromptChain

from flask_api.RAG.config import base_url
import warnings


warnings.filterwarnings("ignore", category=Warning)


llm = Ollama(base_url=base_url, model="qwen2:72b", temperature=0.6)

# 物理链
physics_chain = """
你是一位物理教授，专门负责为用户解决相关的物理问题。当你问题超出了物理的范畴，你会坦率的承认不知道。
问题：{input}
"""

# physics_prompt = PromptTemplate.from_template(physics_chain)

math_chain = """
你是一位数学教授，专门负责为用户解决相关的数学问题。当你问题超出了数学的范畴，你会坦率的承认不知道。
问题：{input}
"""
# math_prompt = PromptTemplate.from_template(math_chain)

chinese_chain = """
你是一位文学教授，专门负责为用户解决相关的文学问题。当你问题超出了文学的范畴，你会坦率的承认不知道。
问题：{input}
"""

# chinese_prompt = PromptTemplate.from_template(chinese_chain)

prompt_infos = [{
    "name": "physice",
    "description": "擅长回答物理问题",
    "prompt_template": physics_chain,
},
    {
        "name": "math",
        "description": "擅长回答数学问题",
        "prompt_template": math_chain,
    }, {
        "name": "chinese",
        "description": "擅长回答文学问题",
        "prompt_template": chinese_chain,
    }]

destination_chains = {}
for prompt_info in prompt_infos:
    name = prompt_info["name"]
    prompt_template = prompt_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])

    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

default_chain = ConversationChain(llm=llm, output_key="text")


destinations = [f"{prompt_info['name']}:{prompt_info['description']}" for prompt_info in prompt_infos]
destinations_str = '\n'.join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)


router_prompt = PromptTemplate(template=router_template, input_variables=["input"],output_parser=RouterOutputParser())

router_chain = LLMRouterChain.from_llm(llm=llm, prompt=router_prompt)
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains = destination_chains,
    default_chain = default_chain,
    verbose=True
)
res = chain.run("汽车的英文怎么写的")
print(res)