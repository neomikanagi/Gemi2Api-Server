import asyncio
import base64
import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from gemini_webapi import GeminiClient, set_log_level
from gemini_webapi.constants import Model
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
set_log_level("INFO")

app = FastAPI(title="Gemini API FastAPI Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global client
gemini_client = None

# Authentication credentials
SECURE_1PSID = os.environ.get("SECURE_1PSID", "")
SECURE_1PSIDTS = os.environ.get("SECURE_1PSIDTS", "")
API_KEY = os.environ.get("API_KEY", "")
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "false").lower() == "true"

# Print debug info at startup
if not SECURE_1PSID or not SECURE_1PSIDTS:
    logger.warning("⚠️ Gemini API credentials are not set or empty! Please check your environment variables.")
else:
    logger.info(f"Credentials found. SECURE_1PSID starts with: {SECURE_1PSID[:5]}...")
    logger.info(f"Credentials found. SECURE_1PSIDTS starts with: {SECURE_1PSIDTS[:5]}...")

if not API_KEY:
    logger.warning("⚠️ API_KEY is not set or empty! API authentication will not work.")
else:
    logger.info(f"API_KEY found. API_KEY starts with: {API_KEY[:5]}...")

def correct_markdown(md_text: str) -> str:
    def simplify_link_target(text_content: str) -> str:
        match_colon_num = re.match(r"([^:]+:\d+)", text_content)
        if match_colon_num:
            return match_colon_num.group(1)
        return text_content

    def replacer(match: re.Match) -> str:
        outer_open_paren = match.group(1)
        display_text = match.group(2)
        new_target_url = simplify_link_target(display_text)
        new_link_segment = f"[{display_text}]({new_target_url})"
        if outer_open_paren:
            return f"{outer_open_paren}{new_link_segment})"
        else:
            return new_link_segment

    pattern = r"(\()?\[([^`]+?)\]\[](https://www.google.com/search\?q=)(.*?)(?<!\\)\)\)*(\))?"
    fixed_google_links = re.sub(pattern, replacer, md_text)
    pattern = r"`(\[[^\]]+\]\([^\)]+\))`"
    return re.sub(pattern, r"\1", fixed_google_links)

# Pydantic models
class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

# Authentication dependency
async def verify_api_key(authorization: str = Header(None)):
    if not API_KEY:
        return  # Skip validation if no API_KEY set
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        if token != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    return token

# Error handler
@app.middleware("http")
async def error_handling(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": {"message": str(e), "type": "internal_server_error"}})

@app.get("/v1/models")
async def list_models():
    now = int(datetime.now(tz=timezone.utc).timestamp())
    data = [
        {
            "id": m.model_name if hasattr(m, "model_name") else str(m),
            "object": "model",
            "created": now,
            "owned_by": "google-gemini-web",
        }
        for m in Model
    ]
    return {"object": "list", "data": data}

def map_model_name(openai_model_name: str) -> Model:
    all_models = [m.model_name if hasattr(m, "model_name") else str(m) for m in Model]
    logger.info(f"Available models: {all_models}")
    for m in Model:
        model_name = m.model_name if hasattr(m, "model_name") else str(m)
        if openai_model_name.lower() in model_name.lower():
            return m
    # Default fallback
    return next(iter(Model))

# Prepare conversation + handle both base64 and uploaded files
async def prepare_conversation_and_files(messages: List[Message], uploaded_files: Optional[List[UploadFile]] = None) -> tuple[str, List[str]]:
    conversation = ""
    temp_files: List[str] = []

    for msg in messages:
        if isinstance(msg.content, str):
            if msg.role == "system":
                conversation += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                conversation += f"Human: {msg.content}\n\n"
            elif msg.role == "assistant":
                conversation += f"Assistant: {msg.content}\n\n"
        else:
            if msg.role == "user":
                conversation += "Human: "
            elif msg.role == "system":
                conversation += "System: "
            elif msg.role == "assistant":
                conversation += "Assistant: "

            for item in msg.content:
                if item.type == "text":
                    conversation += item.text or ""
                elif item.type == "image_url" and item.image_url:
                    image_url = item.image_url.get("url", "")
                    if image_url.startswith("data:image/"):
                        try:
                            base64_data = image_url.split(",")[1]
                            image_data = base64.b64decode(base64_data)
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                            tmp.write(image_data)
                            tmp.close()
                            temp_files.append(tmp.name)
                        except Exception as e:
                            logger.error(f"Base64 image error: {str(e)}")
            conversation += "\n\n"

    # Handle direct uploaded files (multipart)
    if uploaded_files:
        for file in uploaded_files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
            content = await file.read()
            tmp.write(content)
            tmp.close()
            temp_files.append(tmp.name)

    conversation += "Assistant: "
    return conversation, temp_files

async def get_gemini_client():
    global gemini_client
    if gemini_client is None:
        gemini_client = GeminiClient(SECURE_1PSID, SECURE_1PSIDTS)
        await gemini_client.init(timeout=300)
    return gemini_client

@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    files: Optional[List[UploadFile]] = File(None),  # 新增：支持直接文件上传
    api_key: str = Depends(verify_api_key)
):
    try:
        client = await get_gemini_client()
        model = map_model_name(request.model)

        # 处理对话 + 文件（支持 base64 + 直接上传）
        conversation, temp_files = await prepare_conversation_and_files(request.messages, files)

        # 生成响应
        response = await client.generate_content(
            conversation,
            files=temp_files or None,
            model=model
        )

        # 清理临时文件
        for path in temp_files:
            try:
                os.unlink(path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {path}: {str(e)}")

        # 提取回复
        reply_text = ""
        if ENABLE_THINKING and hasattr(response, "thoughts"):
            reply_text += f"<think>{response.thoughts}</think>\n"
        reply_text += response.text if hasattr(response, "text") else str(response)
        reply_text = correct_markdown(reply_text)

        if not reply_text.strip():
            reply_text = "服务器返回空响应，请检查 cookies 是否有效。"

        completion_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        if request.stream:
            async def stream_generator():
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(data)}\n\n"

                for char in reply_text:
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": char}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    await asyncio.sleep(0.01)

                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            result = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created_time,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply_text},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(conversation.split()),
                    "completion_tokens": len(reply_text.split()),
                    "total_tokens": len(conversation.split()) + len(reply_text.split()),
                }
            }
            return result

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "online", "message": "Gemini API Server running with file upload & vision support"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
