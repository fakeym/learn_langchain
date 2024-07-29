from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1 import BaseModel, Field
from pymilvus import MilvusClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv


_ = load_dotenv()


class GraderTool(BaseModel):

    binary_score: str = Field(description="文档与问题的相关性，'yes' or 'no'")


class HallucinationsTool(BaseModel):

    binary_score: str = Field(description="问题与回答的相关性，'yes' or 'no'")


class answerQuestionTool(BaseModel):

    binary_score: str = Field(description="问题与回答的相关性，'yes' or 'no'")

# 1、先封装检索的方法

class gradeAndGenerateRagTool(object):


    def __init__(self):
        self.milvus_client = MilvusClient(host="127.0.0.1", port="19530")
        self.embeding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(temperature=0.5, model="gpt-4o")
        self.struct_llm_grader = self.llm.with_structured_output(GraderTool)
        self.struct_llm_halluciation = self.llm.with_structured_output(HallucinationsTool)
        self.struct_llm_answer = self.llm.with_structured_output(answerQuestionTool)




    def embed_dim(self,text):
        return self.embeding.embed_query(text)

    def search_vector(self,question):
        result = self.milvus_client.search(collection_name="RAG_vector",data=[self.embed_dim(question)],output_fields=["text"])
        return result


    def grade(self, question, text,idx):
        system_grade_prompt = """你是一名评估检索到到文档与用户的问题相关性的评分员，不需要一个严格的测试，目标是过滤掉错误的检索，如果文档包含与用户问题相关的关键字或者语义，请评为相关，否则请评为不相关。你的回答只能是yes或者no。
        """
        grade_message = [SystemMessage(content=system_grade_prompt)]
        grade_message.append(HumanMessage(content=f"问题：{question}\n文档：{text}"))
        result = self.struct_llm_grader.invoke(grade_message)
        return result.binary_score,idx


    def generate(self, question, text):
        generate_human_prompt = f"""
        您是问答任务的助理。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。最多使用三句话，保持答案简洁。\n问题：{question}\n上下文：{text}\n答案：
        """
        human_prompt = ChatPromptTemplate.from_template(generate_human_prompt)
        end_prompt = human_prompt.format_messages(question=question,text=text)
        result = self.llm.invoke(end_prompt)
        return result.content


    def hallucinations(self, documents, answer):
        hallucinations_prompt = "你是一名评估LLM生成是否基于一组检索到的事实的评分员，如果是基于检索得到的事实回答则返回no，否则返回yes"
        hallucinations_message = [SystemMessage(content=hallucinations_prompt)]
        hallucinations_message.append(HumanMessage(content=f"文档：{documents}\n回答：{answer}"))
        result = self.struct_llm_halluciation.invoke(hallucinations_message)
        return result.binary_score


    def answer_question(self, question, answer):
        answer_question_prompt = """你是一名评分员，评估答案是否解决了问题，如果解决了则返回yes，否则返回no"""
        answer_question_message = [SystemMessage(content=answer_question_prompt)]
        answer_question_message.append(HumanMessage(content=f"问题：{question}\n回答：{answer}"))
        result = self.struct_llm_answer.invoke(answer_question_message)
        return result.binary_score


    def rewrite_question(self, question):
        rewrite_question_prompt = """你是一个将输入问题转化为优化的更好版本的问题重写器，用户向量数据库检索，查看输入并尝试推理潜在的语义意图或者含义
        """
        rewrite_messages = [SystemMessage(content=rewrite_question_prompt)]
        rewrite_messages.append(HumanMessage(content=f"问题：{question}"))
        result = self.llm.invoke(rewrite_messages)
        return result.content




