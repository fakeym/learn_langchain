from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pymilvus import MilvusClient

from llm.project.config import base_url

_ = load_dotenv()


class GradeAndGenerateTool(object):

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o")
        self.llm_7b = ChatOpenAI(temperature=0, model="qwen2-instruct", base_url=base_url, api_key="xxx")
        self.embeding = OpenAIEmbeddings(model="bce-embedding-base_v1",base_url=base_url,api_key="xxx")

    # 评分
    def grade(self, question, text, idx):
        system_prompt = """
                你是一名评估检索到到文档与用户到问题相关性到评分员，不需要一个严格的测试，目标是过滤掉错误的检索。
                如果文档包含与用户问题相关的关键字或者语义，请评为相关，否则请评为不相关。
                相关请返回'yes',不相关的请返回'no',你只能回答yes或者no，不能回答其他任何的相关信息
                """
        grade_messages = [SystemMessage(content=system_prompt)]
        grade_messages.append(HumanMessage(content=f"问题：{question}\n文档：{text}"))
        result = self.llm_7b.invoke(grade_messages)
        return result.content.strip(), idx

    # 生成答案
    def generate(self, question, text):
        grade_human_prompt = f"""您是问答任务的助理。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。最多使用三句话，保持答案简洁。\n问题：{question}\n上下文：{text}\n答案："""
        human_prompt = ChatPromptTemplate.from_template(grade_human_prompt)
        grade_human_prompt_end = human_prompt.format_messages(question=question, text=text)
        result = self.llm_7b.invoke(grade_human_prompt_end)
        return result.content

    # 判断是否有幻觉
    def hallucinations(self, documents, answer):
        hallucinations_prompt = """您是一名评估LLM生成是否基于一组检索到的事实的评分员。
        如果回答是基于检索到的信息进行回答的，则返回'no',如果回答不是基于检索到的信息进行回答的，则返回'yes'
        你只能回答yes或者no，不能回答其他任何的相关信息"""
        hallucinations_messages = [SystemMessage(content=hallucinations_prompt)]
        hallucinations_messages.append(HumanMessage(content=f"：回答:{answer}\n文档：{documents}"))
        result = self.llm_7b.invoke(hallucinations_messages)
        return result.content.strip()

    # 判断答案是否和问题相关
    def answer_question(self, question, answer):
        answer_question_prompt = """
                你是一名评分员，评估答案是否解决了问题，如果解决了则返回yes，否则返回no,
                你只能回答yes或者no，不能回答其他任何的相关信息
                """
        answer_question_messages = [SystemMessage(content=answer_question_prompt)]
        answer_question_messages.append(HumanMessage(content=f"问题：{question}\n回答：{answer}"))
        result = self.llm_7b.invoke(answer_question_messages)
        return result.content.strip()

    # 复写问题
    def rewrite_question(self, question):
        rewrite_promtp = "您是一个将输入问题转换为优化的更好版本的问题重写器\n用于矢量库检索。查看输入并尝试推理潜在的语义意图/含义。"
        rewrite_promtp_messages = [SystemMessage(content=rewrite_promtp)]
        rewrite_promtp_messages.append(HumanMessage(content=f"问题：{question}"))
        result = self.llm.invoke(rewrite_promtp_messages)
        return result.content

    def embed_dim(self, text):
        return self.embeding.embed_query(text)

    # 检索
    def search_vector(self, question, collection_name):
        self.milvus_client = MilvusClient(host="127.0.0.1", port="19530")
        result = self.milvus_client.search(collection_name=collection_name, data=[self.embed_dim(question)],
                                           output_fields=["text"])
        self.milvus_client.close()
        return result
