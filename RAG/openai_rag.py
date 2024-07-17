import os
import time
import warnings

import openai
from flask import Flask, request
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from openai import OpenAI

app = Flask(__name__)

warnings.filterwarnings("ignore", category=Warning)
openai.api_key = "sk-3SkxtK7emotHjEvEn6A8FaFOWNo2ycmZNi5qHfgsMIY9SpdF"
openai.api_base = "https://api.fe8.cn/v1"


# client = OpenAI(api_key="sk-3SkxtK7emotHjEvEn6A8FaFOWNo2ycmZNi5qHfgsMIY9SpdF", base_url="https://api.fe8.cn/v1")


class ChatDoc(object):


    def __init__(self, filename):
        self.loader = {
            ".pdf": PyPDFLoader,
            ".txt": Docx2txtLoader,
            ".docx": Docx2txtLoader,
        }
        self.filename = filename
        self.split_text = []

        self.messages = [("system",
                          "你是一位处理文档的工作者，可以根据用户提供的文档内容和上下文内容，用来回答用户的问题。若问题与文档内容不想关，则告诉用户我是一个专注于营地的智能客服，无法回答这个问题，请见谅。文档内容:{context}"),
                         ("human", "你好"),
                         ("assistant", "你好！请问有什么能够帮您？"),
                         ("human", "{question}")]
        self.prompt = ChatPromptTemplate.from_messages(self.messages)
        self.client = OpenAI(api_key="sk-3SkxtK7emotHjEvEn6A8FaFOWNo2ycmZNi5qHfgsMIY9SpdF",
                             base_url="https://api.fe8.cn/v1")

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
            split_text = text_splitter.split_documents(full_text)
            self.split_text = split_text

    def vector_storage(self):
        fs = LocalFileStore("cache/")
        start_time = time.time()
        emmbeddings = OpenAIEmbeddings(openai_api_key="sk-3SkxtK7emotHjEvEn6A8FaFOWNo2ycmZNi5qHfgsMIY9SpdF",
                                       openai_api_base="https://api.fe8.cn/v1")
        cache_embedding = CacheBackedEmbeddings.from_bytes_store(emmbeddings, fs, namespace=emmbeddings.model)
        db = FAISS.from_documents(documents=self.split_text, embedding=cache_embedding)
        print(time.time() - start_time)
        return db

    def ask_and_find(self, question):
        db = self.vector_storage()
        start_time = time.time()
        retriever = db.as_retriever(search_kwargs={'k': 3})
        result = retriever.get_relevant_documents(question)
        print("检索耗时：", time.time() - start_time)
        return result

    def chat_with_doc(self, question):
        # self.messages.append(("human", f"{question}"))
        self.prompt = ChatPromptTemplate.from_messages(messages=self.messages)
        _content = ""
        contexts = self.ask_and_find(question)
        for context in contexts:
            _content += context.page_content
        messages = self.prompt.format_messages(context=_content, question=question)
        res = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key="sk-3SkxtK7emotHjEvEn6A8FaFOWNo2ycmZNi5qHfgsMIY9SpdF",
            base_url="https://api.fe8.cn/v1",
        )
        result = res.stream(messages)
        content = ""
        for i in result:
            content += i.content
        # self.messages.append(("assistant", f"{content}"))
        return content


chat_doc = ChatDoc("aa.docx")
chat_doc.split_sentences()


@app.route("/question", methods=["POST"])
def question_answer():
    question = request.form["question"]
    content = chat_doc.chat_with_doc(question)
    # for i in content:
    #     yield i.content
    # loop = asyncio.get_event_loop()
    # content = loop.run_until_complete(chat_doc.chat_with_doc(question))
    # print(content)
    # for i in content:
    #     print(i)
    #     yield i
    return content


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
# if __name__ == '__main__':
#
#     chat_doc = ChatDoc("aa.docx")
#     chat_doc.split_sentences()
#     while True:
#         question = input("请输入你的问题：")
#         start_time = time.time()
#         a = chat_doc.chat_with_doc(question)
#         print(a)
#         print(time.time() - start_time)
