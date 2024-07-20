"""
1、rag的应用场景：比如内部非公开的数据，进行知识检索。领域的智能客服。。。
2、流程：
    1、文档加载器  ---主要用于加载对应格式的文档。document对象，字符串类型
    2、文档的拆分器   --- 主要用于对已经加载文档进行一个拆分。
    3、向量化  ---主要用于对已经拆分后的文档进行向量化。
    4、灌库   ---把已经向量化的文本进行灌库，插入到向量数据库中
    5、检索   ---根据用户的问题或者指令，进行对应的信息检索。
"""


import os


os.environ["OPENAI_API_BASE"] = "https://api.fe8.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-nA4XFQzD7IZc8fVTcLDFqH1ds9ySyS39hpl46eOxiTltIfph"

# 1、文档加载

from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("小米SU7.docx")
docs = loader.load()

# 2、文档的拆分

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=240,
    chunk_overlap=30,
    length_function=len,
    add_start_index= True
)
doc_splitter = text_splitter.split_documents(docs)
txt_splitter = text_splitter.split_text(docs[0].page_content)

# 3、向量化

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

vector_list = []
embedding = OpenAIEmbeddings(model = "text-embedding-3-small")
for txt in doc_splitter:
    embedding_dim = embedding.embed_query(txt.page_content)
    vector_list.append({"id":txt.metadata["start_index"],"vector":embedding_dim,"text":txt.page_content})


# 4、灌库
from pymilvus import  MilvusClient
milvus_cli = MilvusClient(host="127.0.0.1", port="19530")
milvus_cli.create_collection(
    collection_name="test_rag",
    dimension=1536,
    metric_type="IP",
    consistency_level="Strong"
)
milvus_cli.insert(collection_name="test_rag",data=vector_list)


# 5、检索
question = "小米SU7的续航怎么样"
embedding_dim = embedding.embed_query(question)
search_result = milvus_cli.search(collection_name="test_rag",data=[embedding_dim],limit=5,output_fields=["text"])
content = ""
for info in search_result[0]:
    content += info["entity"]["text"]
llm = ChatOpenAI(model="gpt-4o",temperature=0)
res = llm.invoke(f"问题：{question},检索的内容是：{content},请简要回答")
print(res.content)