import os.path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from tqdm import tqdm

_ = load_dotenv()

from langchain_community.document_loaders import Docx2txtLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from pymilvus import MilvusClient


class ragTool(object):

    def __init__(self):
        self.loader = {
            ".txt": Docx2txtLoader,
            ".docx": Docx2txtLoader,
            ".csv": CSVLoader,
        }
        self.milvus_client = MilvusClient(host="127.0.0.1", port="19530")
        self.llm = ChatOpenAI(model="gpt-4o")
        self.messages = [SystemMessage(
            content="你是一个助手，请根据上下文回答问题，如果无法回答，请说“我不理解”，请尽量简要回答，与问题不相关的内容不用作为分析的内容。")]

    def get_file(self, filename):
        """
        获取文件
        :param filename: 文件名
        :return:
        """
        file_type = os.path.splitext(filename)[-1]
        if file_type in self.loader:
            loader = self.loader[file_type]
            loader = loader(filename)
            return loader.load()
        else:
            return None

    def split_sentences(self, filename):
        """
        将文件分割成句子
        :param filename: 文件名
        :return:
        """
        full_text = self.get_file(filename)
        if full_text:
            text_splitter = CharacterTextSplitter(chunk_size=240, chunk_overlap=30, add_start_index=True,
                                                  length_function=len)
            text_split = text_splitter.split_documents(full_text)
            return text_split
        else:
            return "文档格式不支持"

    def emb_text(self, text):
        """
        将文本向量化
        :param text: 文本
        :return:
        """
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return embeddings.embed_query(text)

    def vector_storage(self, filename):
        text_split = self.split_sentences(filename)
        data_vector = []
        for idx, text in enumerate(tqdm(text_split, desc="Embedding")):
            data_vector.append({
                "id": idx,
                "text": text.page_content,
                "vector": self.emb_text(text.page_content)
            })

        self.milvus_client.create_collection(
            collection_name="test_collection",
            dimension=1536,
            metric_type="IP",
            consistency_level="Strong"
        )

        self.milvus_client.insert(collection_name="test_collection", data=data_vector)
        return "success"

    def query_data(self, query):
        query_vector = self.emb_text(query)
        result = self.milvus_client.search(
            collection_name="test_collection",
            data=[query_vector],
            limit=3,
            output_fields=["text"],
            params={"metric_type": "IP"},
        )

        result_info = ""
        for info in result[0]:
            result_info += info["entity"]["text"]

        return result_info

    def get_answer(self, question):
        """
        获取答案
        :param question: 问题
        :return:
        """

        result = self.query_data(question)

        self.messages.append(HumanMessage(content=f"问题:{question},检索内容:{result}"))
        res = self.llm.invoke(self.messages)
        self.messages.append(AIMessage(content=res.content))
        return res


