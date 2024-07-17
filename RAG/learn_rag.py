from langchain_community.llms import Ollama
from config import base_url
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.storage import LocalFileStore
import time


start_time = time.time()

ollama = Ollama(base_url=base_url,model="qwen2:72b")
# print(ollama.invoke(""))
loader = Docx2txtLoader("aa.docx")
text = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50,length_function=len,add_start_index=True)

embedding = OllamaEmbeddings(base_url=base_url,model="mxbai-embed-large")

fs = LocalFileStore("../cache/")
cache_embedding = CacheBackedEmbeddings.from_bytes_store(embedding,fs, namespace=embedding.model)
print(list(fs.yield_keys()))
data = text_splitter.split_text(text=text[0].page_content)
#
db = chroma.Chroma.from_texts(texts=data,embedding=cache_embedding)

print(list(fs.yield_keys()))
retriver = db.as_retriever()

template = """
{context}
{question}
"""


prompt = ChatPromptTemplate.from_template(template=template)

chain = {"context":retriver,"question":RunnablePassthrough()} | prompt | ollama | StrOutputParser()


res = chain.invoke("这家公司叫什么名字")
print(res)
end_time = time.time()
print(end_time - start_time)










