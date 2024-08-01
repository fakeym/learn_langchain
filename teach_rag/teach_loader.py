import json
import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter, RecursiveJsonSplitter
from langchain_community.document_loaders import Docx2txtLoader, CSVLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pymilvus import MilvusClient
from tqdm import tqdm

_ = load_dotenv()


# 1、首先先初始化文档加载器，在使用load方法进行加载
# 2、初始化文本拆分器，再进行拆分

class ChatDoc(object):

    def __init__(self):
        self.loader = {
            ".pdf": PyPDFLoader,
            ".txt": Docx2txtLoader,
            ".docx": Docx2txtLoader,
            ".md": UnstructuredMarkdownLoader,
            ".csv": CSVLoader,
            ".json": self.handle_json,
        }

        self.txt_splitter = CharacterTextSplitter(chunk_size=240, chunk_overlap=30, length_function=len,
                                                  add_start_index=True)
        self.json_splitter = RecursiveJsonSplitter(max_chunk_size=240)
        self.embeding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.milvus_client = MilvusClient(host="127.0.0.1", port="19530")
        self.messages = [SystemMessage(
            content="你是一位处理文档的工作者，可以根据用户提供的文档内容和上下文内容，用来回答用户的问题。若问题与文档内容不想关，你的回答尽可能的简洁，不需要过多的阐述")]
        self.llm = ChatOpenAI(temperature=0.5,model="gpt-4o")

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

        self.milvus_client.create_collection(collection_name="rag", dimension=1536, metric_type="IP",
                                             consistency_level="Strong")
        self.milvus_client.insert(collection_name="rag", data=data)
        return "向量存储成功"

    def ask_and_find(self, question):
        search_result = self.milvus_client.search(collection_name="rag", data=[self.emb_text(question)], limit=3,
                                                  params={"metric_type": "IP"}, output_fields=["text"])

        content = ""
        for info in search_result[0]:
            content += info["entity"]["text"]
        return content

    def chat_with_doc(self, question):
        content = self.ask_and_find(question)
        self.messages.append(HumanMessage(content=f"问题：{question},内容：{content}"))
        prompt = ChatPromptTemplate.from_messages(self.messages)
        messages = prompt.format_messages(question=question, content=content)
        res = self.llm.stream(messages)
        context = ""
        for i in res:
            context += i.content

        self.messages.append(AIMessage(content=context))
        return context


if __name__ == '__main__':
    chatdoc = ChatDoc()
    # chatdoc.split_text("test.docx")
    # chatdoc.vector_storage()
    res = chatdoc.chat_with_doc("该影片在北美的多少家影院试映 ")
    print(res)
