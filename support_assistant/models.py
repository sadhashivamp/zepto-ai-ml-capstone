from pydantic import BaseModel


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float