from fastapi import FastAPI, File, UploadFile,Query

from pydantic import BaseModel, Field

from llm.project.create_graph_teach import createGraph

graph = createGraph()

app = FastAPI()


class questionInput(BaseModel):
    question: str = Field(..., description="问题")
#     file :UploadFile=File(description="文件")


@app.post("/chat_file")
async def chat_file(question:str=Query(None),file:UploadFile=File(None)):
    graph_run = graph.create_graph()
    if file:
        contents = await file.read()
        file_name = file.filename
        with open(file_name, "wb") as f:
            f.write(contents)
        data = {"question": question, "filename": file_name}
    else:
        data = {"question": question,"filename":None}

    res = graph_run.invoke(data)
    return res["answer"]



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=8000)