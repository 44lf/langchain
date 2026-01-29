from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.utils.llm_client import LLMClient

class LLMRouter:
    router = APIRouter()
    @router.get("/ask")
    async def llm_ask(prompt: str):
        return LLMClient().ask(prompt)

    @router.get("/stream_ask")
    async def llm_stream_ask(prompt: str):
        client = LLMClient()

        def event_gen():
            yield "event: start\ndata:ok\n\n"
            for text in client.stream_ask(prompt):
                text = text.replace("\n", " ")
                for line in text.split("\n"):
                    yield f"data: {line}\n\n"
                yield"\n"
            yield "event: end\ndata:done\n\n"    
        return StreamingResponse(event_gen(), media_type="text/event-stream")

