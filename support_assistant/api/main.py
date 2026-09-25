from fastapi import FastAPI
from pydantic import BaseModel

from support_assistant.graph import graph, make_response


app = FastAPI()


class AskRequest(BaseModel):
    query: str


@app.post("/ask")
def ask(request: AskRequest):
    result = graph.invoke({
        "query": request.query
    })

    response = make_response(result)

    return response