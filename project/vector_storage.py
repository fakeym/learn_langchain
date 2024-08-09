import json
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader,TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveJsonSplitter
_ = load_dotenv("/Users/zhulang/work/llm/self_rag/.env")


class VectorStorageObject(object):

    def __init__(self):
        self.loader = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
            ".csv": CSVLoader,
            ".json": self.handle_json,
        }

        self.txt_splitter = CharacterTextSplitter(chunk_size=240, chunk_overlap=30, length_function=len,
                                                  add_start_index=True)
        self.json_splitter = RecursiveJsonSplitter(max_chunk_size=240)

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
                self.end_splitter = [doc.page_content for doc in self.end_splitter]

            return self.end_splitter

        else:
            raise "文件格式不支持"


