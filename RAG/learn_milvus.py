import os
import time
import warnings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from pymilvus import MilvusClient
from tqdm import tqdm
import redis

from config import OPENAI_BASE_URL

warnings.filterwarnings("ignore", category=Warning)

# os.environ["OPENAI_API_BASE"] = "https://api.fe8.cn/v1"
os.environ["OPENAI_API_BASE"] = OPENAI_BASE_URL
os.environ["OPENAI_API_KEY"] = "sk-ucOibt6QEWOo3zM4YFNfeUv9eyApPysO2ZFgMEtE3K252pxt"


# os.environ["OPENAI_BASE_URL"] = "http://ip:port/v1"

class ChatDoc(object):

    def __init__(self, filename):
        self.loader = {
            ".pdf": PyPDFLoader,
            ".txt": Docx2txtLoader,
            ".docx": Docx2txtLoader,
        }
        self.filename = filename
        self.split_text = []

        self.messages = [SystemMessage(content="你是一位处理文档的工作者，可以根据用户提供的文档内容和上下文内容，用来回答用户的问题。若问题与文档内容不想关，则告诉用户我是一个专注于营地的智能客服，无法回答这个问题，请见谅。文档内容:{context}"),
                         HumanMessage(content="你好"),
                         AIMessage(content="你好！请问有什么能够帮您？")]

        self.milvus_client = MilvusClient(host="127.0.0.1", port="19530")
        self.r = redis.Redis(host="127.0.0.1", port=6379, db=0)

    def get_file(self):
        file_extension = os.path.splitext(self.filename)[1]
        loader_class = self.loader.get(file_extension, None)
        if loader_class:
            data = loader_class(self.filename).load()
            return data
        else:
            return None

    def split_sentences(self):
        full_text = self.get_file()
        if full_text:
            text_splitter = CharacterTextSplitter(chunk_size=210, chunk_overlap=30, add_start_index=True,
                                                  length_function=len)
            split_text = text_splitter.split_text(full_text[0].page_content)
            self.split_text = split_text

    def emb_text(self, text):
        embedding = OpenAIEmbeddings(model="text-embedding-3-small", base_url="https://api.fe8.cn/v1")
        embedding_dim = embedding.embed_query(text)
        return embedding_dim

    def vector_storage(self):
        data_vector = []
        for idx, text in enumerate(tqdm(self.split_text, desc="测试向量数据库")):
            data_vector.append({"id": idx, "text": text, "vector": self.emb_text(text)})

        self.milvus_client.create_collection(
            collection_name="RAG_vector",
            dimension=1536,
            metric_type="IP",  # Inner product distance
            consistency_level="Strong",  # Strong consistency level
        )
        self.milvus_client.insert(collection_name="RAG_vector", data=data_vector)

    def ask_and_find(self, question):
        search_result = self.milvus_client.search(
            collection_name="RAG_vector",
            data=[self.emb_text(question)],
            limit=3,
            params={"metric_type": "IP"},
            output_fields=["text"],
        )
        info_result = ""
        for res in search_result[0]:
            info_result += res["entity"]["text"]
        return info_result

    def chat_with_doc(self, question):
        self.messages.append(HumanMessage(content=f"{question}"))
        self.prompt = ChatPromptTemplate.from_messages(self.messages)
        contexts = self.ask_and_find(question)
        messages = self.prompt.format_messages(context=contexts, question=question)
        llm = ChatOpenAI(
            temperature=0,
            model="Qwen2-14B-merge-GPTQ-Int8",
        )
        res = llm.stream(messages)
        content = ""
        for i in res:
            content += i.content
        self.messages.append(AIMessage(content=f"{content}"))
        return content


if __name__ == '__main__':
    chat_doc = ChatDoc("aa.docx")
    chat_doc.split_sentences()
    chat_doc.vector_storage()
    while True:
        question = input("请输入你的问题：")
        start_time = time.time()
        a = chat_doc.chat_with_doc(question)
        print(a)
        print(time.time() - start_time)

