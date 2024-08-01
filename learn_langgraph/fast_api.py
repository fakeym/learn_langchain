from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field
from starlette.websockets import WebSocket

from flask_api.learn_langGraph.learn_sub_graph import runGraph

from rag_tool_graph import ChatDoc

chat = ChatDoc()
langgraph = runGraph()

app = FastAPI()


class RagInput(BaseModel):
    question: str = Field(description="问题")


@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename
    with open(filename, 'wb') as f:
        f.write(contents)
    chat.get_file(filename)
    chat.split_sentences(filename)
    chat.vector_storage()
    return "success"


@app.post("/question")
async def search(question: RagInput):
    result = chat.chat_with_doc(question.question)
    return {"result": result}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        result = langgraph.run_langgraph(question=data)
        await websocket.send_text(result)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
