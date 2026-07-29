"""FastAPI application entry point: router registration and startup lifespan."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_cnn_model, get_pinecone_index
from api.routes import auth, image, text, feedback, history, conversation, agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the CNN model and Pinecone index once at server startup, not on first request."""
    get_cnn_model()
    get_pinecone_index()
    yield


app = FastAPI(title="Thoragrid API", lifespan=lifespan)

# Tighten allow_origins to the actual Streamlit origin before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, tags=["auth"])
app.include_router(image.router, tags=["image"])
app.include_router(text.router, tags=["text"])
app.include_router(feedback.router, tags=["feedback"])
app.include_router(history.router, tags=["history"])
app.include_router(conversation.router, tags=["conversation"])
app.include_router(agent.router, tags=["agent"])


@app.get("/health")
def health_check() -> dict:
    """Basic liveness check."""
    return {"status": "ok"}