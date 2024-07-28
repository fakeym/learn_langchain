from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic.v1 import BaseModel, Field
from pymilvus import MilvusClient

_ = load_dotenv()


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


class GradeAndGenerateTool(object):

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.5, model="gpt-4o")
        self.struct_llm_grader = self.llm.with_structured_output(GradedRagTool)
        self.struct_llm_hallucinations = self.llm.with_structured_output(GradeHallucinations)
        self.struct_llm_answer = self.llm.with_structured_output(GradeAnswer)
        self.embeding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.milvus_client = MilvusClient(host="127.0.0.1", port="19530")
        self.milvus_client.create_collection(collection_name="rag", dimension=1536, metric_type="IP",
                                             consistency_level="Strong")

    # 评分
    def grade(self, question, text):
        system_prompt = """
                你是一名评估检索到到文档与用户到问题相关性到评分员，不需要一个严格的测试，目标是过滤掉错误的检索。如果文档包含与用户问题相关的关键字或者语义，请评为相关，否则请评为不相关。你的回答只能是yes或者no
                """
        grade_messages = [SystemMessage(content=system_prompt)]
        grade_messages.append(HumanMessage(content=f"问题：{question}\n文档：{text}"))
        result = self.struct_llm_grader.invoke(grade_messages)
        return result.binary_score

    # 生成答案
    def generate(self, question, text):
        grade_human_prompt = f"""您是问答任务的助理。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。最多使用三句话，保持答案简洁。\n问题：{question}\n上下文：{text}\n答案："""
        human_prompt = ChatPromptTemplate.from_template(grade_human_prompt)
        grade_human_prompt_end = human_prompt.format_messages(question=question, text=text)
        result = self.llm.invoke(grade_human_prompt_end)
        return result.content

    # 判断是否有幻觉
    def hallucinations(self, documents, answer):
        hallucinations_prompt = "您是一名评估LLM生成是否基于一组检索到的事实的评分员。如果是基于检索到的事实回答则返回no，否则返回yes"
        hallucinations_messages = [SystemMessage(content=hallucinations_prompt)]
        hallucinations_messages.append(HumanMessage(content=f"：回答:{answer}\n文档：{documents}"))
        result = self.struct_llm_hallucinations.invoke(hallucinations_messages)
        return result.binary_score

    # 判断答案是否和问题相关
    def answer_question(self, question, answer):
        answer_question_prompt = """
                你是一名评分员，评估答案是否解决了问题，如果解决了则返回yes，否则返回no
                """
        answer_question_messages = [SystemMessage(content=answer_question_prompt)]
        answer_question_messages.append(HumanMessage(content=f"问题：{question}\n回答：{answer}"))
        result = self.struct_llm_answer.invoke(answer_question_messages)
        return result.binary_score

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
    def search_vector(self, question):
        result = self.milvus_client.search(collection_name="RAG_vector", data=[self.embed_dim(question)],
                                           output_fields=["text"])
        return result


if __name__ == '__main__':
    tools = GradeAndGenerateTool()
    question = "你们的地址在哪里"
    print(tools.search_vector(question)[0])


