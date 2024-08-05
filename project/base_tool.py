import json
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader, CSVLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter, RecursiveJsonSplitter
from pymilvus import MilvusClient
from tqdm import tqdm

from llm.project.base_model import RouteQuery

_ = load_dotenv("/Users/zhulang/work/llm/self_rag/.env")


# 灌库
class ChatDoc(object):

    def __init__(self, collection_name):
        self.loader = {
            ".pdf": PyPDFLoader,
            ".txt": Docx2txtLoader,
            ".docx": Docx2txtLoader,
            ".md": UnstructuredMarkdownLoader,
            ".csv": CSVLoader,
            ".json": self.handle_json,
        }
        self.collection_name = collection_name

        self.txt_splitter = CharacterTextSplitter(chunk_size=240, chunk_overlap=30, length_function=len,
                                                  add_start_index=True)
        self.json_splitter = RecursiveJsonSplitter(max_chunk_size=240)
        self.embeding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.milvus_client = MilvusClient(host="127.0.0.1", port="19530")
        self.milvus_client.create_collection(collection_name=self.collection_name, dimension=1536, metric_type="IP",
                                             consistency_level="Strong")
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o")

    def get_file(self, filename):
        file_extension = os.path.splitext(filename)[-1]
        loader = self.loader.get(file_extension, None)
        if loader:
            if file_extension == ".json":
                return loader(filename)
            else:
                load_info = loader(filename).load()
                return load_info

        else:
            return None

    def handle_json(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = f.read()
        return data

    def is_json(self, data):
        try:
            json.loads(data)
            return True
        except:
            return False

    def split_text(self, filename):
        load_info = self.get_file(filename)
        if load_info:
            if self.is_json(load_info):
                self.end_splitter = self.json_splitter.split_text(json.loads(load_info), ensure_ascii=False)
            else:
                self.end_splitter = self.txt_splitter.split_documents(load_info)

            return self.end_splitter

        else:
            return "文件格式不支持"

    def emb_text(self, text):
        return self.embeding.embed_query(text)

    def vector_storage(self):
        data = []
        for idx, text in enumerate(tqdm(self.end_splitter, desc="向量化")):
            if isinstance(text, Document):
                text = text.page_content
            data.append({"id": idx, "vector": self.emb_text(text), "text": text})

        self.milvus_client.insert(collection_name=self.collection_name, data=data)
        return "向量存储成功"


# 判断文件知识的类型
class determine_type(object):
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o")
        self.struct_llm = self.llm.with_structured_output(RouteQuery)

    # 这个是判断知识应该保存在哪个向量数据库中的那张表
    def vector_tool(self, filename):
        loader = Docx2txtLoader(filename).load()
        content = loader[0].page_content
        res = self.struct_llm.invoke(content)
        vector_storage_tool = ChatDoc(res.route)
        vector_storage_tool.split_text(filename)
        result = vector_storage_tool.vector_storage()
        return result

    def search_tool(self, question):
        res = self.struct_llm.invoke(question)
        print(res.route)
        search_tool = ChatDoc(res.route)
        result = search_tool.chat_with_doc(question)
        print(result)
