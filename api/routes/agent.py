"""POST /agent/query — natural language analytics question via the read-only SQL agent."""

import time
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DBSession

from api.middleware.auth import require_role, TokenPayload
from api.schemas.requests import SqlAgentRequest
from api.schemas.responses import SqlAgentResponse
from scripts.db_session import get_db
from scripts.db_models import SqlAgentLog
from core.sql_agent.agent import run_sql_agent

router = APIRouter()


@router.post("/agent/query", response_model=SqlAgentResponse)
def agent_query(
    body: SqlAgentRequest,
    user: Annotated[TokenPayload, Depends(require_role("admin", "doctor"))],
    db: Annotated[DBSession, Depends(get_db)],
) -> SqlAgentResponse:
    """Run a natural language analytics question through the SQL agent and log it for audit."""
    start = time.perf_counter()

    result = run_sql_agent(question=body.question, role=user.role, doctor_id=user.sub)

    db.add(SqlAgentLog(
        user_id=user.sub, role=user.role, question=body.question,
        sql_executed=result["sql_executed"], row_count=result["row_count"],
    ))
    db.commit()

    return SqlAgentResponse(
        sql_executed=result["sql_executed"],
        answer=result["answer"],
        explanation=result["explanation"],
        rows=result["rows"],
        row_count=result["row_count"],
        latency_ms=int((time.perf_counter() - start) * 1000),
    )