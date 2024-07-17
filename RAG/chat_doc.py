import os
from langchain_community.llms import Ollama
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import ChatPromptTemplate
from config import base_url
from langchain.storage import LocalFileStore
import warnings

warnings.filterwarnings("ignore", category=Warning)


class ChatDoc(object):

    def __init__(self,filename):
        self.loader = {
            ".pdf": PyPDFLoader,
            ".txt": Docx2txtLoader,
            ".docx": Docx2txtLoader
        }

        self.filename = filename
        self.splite_text = []
        self.ollama = Ollama(base_url=base_url,model="qwen2:72b",temperature=0.6)
        self.messages = [("system","你是一位处理文档的工作者，可以根据用户提供的文档内容或者上下文内容，用来回答用户的问题。文档内容:{context}"),
                         ("human","你好"),
                         ("assistant","你好！，请问有什么可以帮您？")]


    def get_file(self):
        file_extension = os.path.splitext(self.filename)[1]
        loader_class = self.loader.get(file_extension,None)
        if loader_class:
            data = loader_class(self.filename).load()
            return data
        else:
            return None


    def split_sentence(self):
        full_text = self.get_file()
        if full_text:
            text_splitter = CharacterTextSplitter(chunk_size=210,chunk_overlap=30,add_start_index=True,length_function=len)
            splite_text = text_splitter.split_documents(full_text)
            self.splite_text = splite_text


    def vector_storage(self):
        fs = LocalFileStore("./cache/")
        embeddings = OllamaEmbeddings(base_url=base_url,model="mxbai-embed-large")
        cache_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings,fs,namespace=embeddings.model)
        db = Chroma.from_documents(documents=self.splite_text,embedding=cache_embeddings)
        return db


    def ask_and_find(self,question):
        db = self.vector_storage()
        retriever = db.as_retriever()
        compressor = LLMChainExtractor.from_llm(llm=self.ollama)
        compressor_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
        result = compressor_retriever.get_relevant_documents(query=question)
        return result

    def chat_with_doc(self,question):
        contexts = self.ask_and_find(question)
        self.messages.append(("human",f"{question}"))
        self.prompt = ChatPromptTemplate.from_messages(self.messages)
        _context = ""
        for context in contexts:
            _context += context.page_content
        print(_context)
        messages = self.prompt.format_messages(context=_context,question=question)
        res = self.ollama.stream(messages)
        res_str = ''.join([i for i in res])
        self.messages.append(("assistant",res_str))
        print(self.messages)
        return res






if __name__ == '__main__':
    chat = ChatDoc("aa.docx")
    chat.split_sentence()
    while True:
        question = input("请输入您的问题：")
        res = chat.chat_with_doc(question)
        for i in res:
            print(i,end="",flush=False)



