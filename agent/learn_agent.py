import time


from flask import Flask, Response
from langchain_openai import ChatOpenAI

app = Flask(__name__)

# 初始化LLMChain，这里使用的是基于OpenAI的模型，需要一个OpenAI的API密钥

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key="sk-u8dqwXbP6SRnpYkna4HcCVpm6JAHEuLc28cXOuHSfI56PqGP",
    base_url="https://api.fe8.cn/v1"
)


@app.route('/stream')
def stream_response():
    response = llm.stream("请帮我制定一份关于学习python的学习计划")

    def generate_stream():
        for i in response:
            yield i.content

    # 返回流式响应
    return Response(generate_stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)
