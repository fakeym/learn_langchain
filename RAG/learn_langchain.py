from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import base_url
import warnings

warnings.filterwarnings("ignore")

ollama = Ollama(base_url=base_url, model="qwen2:72b")

# 加载文档
loader = Docx2txtLoader("aa.docx")
text = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=5, length_function=len,
                                               add_start_index=True)
data = text_splitter.split_text(text=text[0].page_content)
print(data)

vector_database = chroma.Chroma.from_texts(texts=data,
                                           embedding=OllamaEmbeddings(base_url=base_url,
                                                                      model="mxbai-embed-large:latest"))
retriever = vector_database.as_retriever()
print(retriever)
template = """
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | ollama
        | StrOutputParser())

res = chain.invoke("今天是几月几号")
print(res)
a = ollama.invoke("你的知识是到哪天截止")
print(a)
