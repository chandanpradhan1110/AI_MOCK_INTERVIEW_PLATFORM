"""
FastAPI Main Application
Provides REST API for the AI Interview Platform.
"""
import uuid
import io
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

from workflows.interview_graph import InterviewGraph
from rag.retriever import get_retriever, clear_retriever, FAISSRetriever
from resume.parser import ResumeParser
from database.models import get_db, InterviewSession
from config import settings


# ─────────────────────────────────────────────
# Application State
# ─────────────────────────────────────────────

# In-memory session store for interview states
# In production, use Redis for distributed sessions
_session_states: Dict[str, Any] = {}
_session_graphs: Dict[str, InterviewGraph] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("🚀 AI Interview Platform starting up...")
    # Initialize DB connection
    db = get_db()
    logger.info("✅ Application ready")
    yield
    logger.info("🛑 Application shutting down...")
    db.close()


app = FastAPI(
    title="AI Mock Interview Platform",
    description="Production-grade Multi-Agent AI Interview System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────

class StartInterviewRequest(BaseModel):
    role: str = Field(..., description="Target role: Data Scientist | ML Engineer | Python Developer")
    candidate_name: str = Field(default="Candidate")
    job_description: Optional[str] = Field(default=None, description="Job description text")
    max_questions: Optional[int] = Field(default=None, ge=3, le=20)


class SubmitAnswerRequest(BaseModel):
    session_id: str
    answer: str = Field(..., min_length=1, max_length=5000)


class SessionResponse(BaseModel):
    session_id: str
    message: str
    data: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "active_sessions": len(_session_states),
    }


# ─────────────────────────────────────────────
# Session Management
# ─────────────────────────────────────────────

@app.post("/api/v1/interview/start", response_model=SessionResponse)
async def start_interview(request: StartInterviewRequest):
    """
    Start a new interview session.
    Returns session_id and first question.
    """
    session_id = str(uuid.uuid4())
    db = get_db()

    logger.info(f"Starting interview — Role: {request.role} | Session: {session_id}")

    # Initialize retriever for this session
    retriever = get_retriever(session_id)

    # Index JD if provided
    if request.job_description:
        retriever.add_documents(request.job_description, source="job_description")
        logger.info(f"JD indexed for session: {session_id}")

    # Initialize graph
    graph = InterviewGraph(retriever=retriever)
    _session_graphs[session_id] = graph

    # Initialize state
    state = graph.initialize_state(
        session_id=session_id,
        role=request.role,
        candidate_name=request.candidate_name,
        max_questions=request.max_questions,
    )

    # Generate first question
    question = graph.question_agent.generate_opening_question(
        role=request.role,
        candidate_name=request.candidate_name,
    )
    state["current_question"] = question
    state["previous_questions"] = [question]
    state["waiting_for_answer"] = True

    # Save state
    _session_states[session_id] = state

    # Save to DB
    session_doc = InterviewSession.create_document(
        role=request.role,
        candidate_name=request.candidate_name,
        has_resume=False,
        has_jd=bool(request.job_description),
    )
    session_doc["session_id"] = session_id
    db.create_session(session_doc)

    return SessionResponse(
        session_id=session_id,
        message="Interview started successfully",
        data={
            "question": question,
            "question_number": 1,
            "total_questions": state["max_questions"],
            "difficulty": "medium",
            "is_follow_up": False,
            "role": request.role,
        }
    )


@app.post("/api/v1/interview/start-with-resume", response_model=SessionResponse)
async def start_interview_with_resume(
    role: str = Form(...),
    candidate_name: str = Form(default="Candidate"),
    max_questions: Optional[int] = Form(default=None),
    resume: Optional[UploadFile] = File(default=None),
    jd_file: Optional[UploadFile] = File(default=None),
    job_description: Optional[str] = Form(default=None),
):
    """
    Start interview with optional resume PDF and/or JD.
    Supports multipart form upload.
    """
    session_id = str(uuid.uuid4())
    db = get_db()

    logger.info(f"Starting personalized interview — Role: {role} | Session: {session_id}")

    # Parse resume
    resume_context = ""
    resume_data = {}
    has_resume = False

    if resume and resume.filename:
        try:
            pdf_bytes = await resume.read()
            parser = ResumeParser()
            resume_data = parser.parse_from_bytes(pdf_bytes)
            resume_context = parser.get_context()
            has_resume = True
            candidate_name = resume_data.get("name", candidate_name)
            logger.info(f"Resume parsed for: {candidate_name}")
        except Exception as e:
            logger.error(f"Resume parsing error: {e}")
            resume_context = "Resume provided but could not be parsed."

    # Initialize retriever
    retriever = get_retriever(session_id)
    has_jd = False

    # Index JD (from text or PDF)
    jd_text = job_description or ""

    if jd_file and jd_file.filename:
        try:
            from resume.parser import extract_text_from_bytes
            jd_bytes = await jd_file.read()
            jd_text = extract_text_from_bytes(jd_bytes, jd_file.filename)
        except Exception as e:
            logger.error(f"JD file parsing error: {e}")

    if jd_text:
        retriever.add_documents(jd_text, source="job_description")
        has_jd = True
        logger.info(f"JD indexed ({len(jd_text)} chars) for session: {session_id}")

    # Initialize graph
    graph = InterviewGraph(retriever=retriever)
    _session_graphs[session_id] = graph

    # Initialize state
    state = graph.initialize_state(
        session_id=session_id,
        role=role,
        candidate_name=candidate_name,
        resume_context=resume_context,
        resume_data=resume_data,
        max_questions=max_questions,
    )

    # Generate personalized opening
    question = graph.question_agent.generate_opening_question(
        role=role,
        candidate_name=candidate_name,
        resume_context=resume_context,
    )
    state["current_question"] = question
    state["previous_questions"] = [question]
    state["waiting_for_answer"] = True

    _session_states[session_id] = state

    # Save to DB
    session_doc = InterviewSession.create_document(
        role=role,
        candidate_name=candidate_name,
        has_resume=has_resume,
        has_jd=has_jd,
    )
    session_doc["session_id"] = session_id
    db.create_session(session_doc)

    return SessionResponse(
        session_id=session_id,
        message="Personalized interview started",
        data={
            "question": question,
            "question_number": 1,
            "total_questions": state["max_questions"],
            "difficulty": "medium",
            "is_follow_up": False,
            "role": role,
            "candidate_name": candidate_name,
            "has_resume": has_resume,
            "has_jd": has_jd,
            "resume_skills": resume_data.get("skills", []) if resume_data else [],
        }
    )


@app.post("/api/v1/interview/answer", response_model=SessionResponse)
async def submit_answer(request: SubmitAnswerRequest):
    """
    Submit an answer to the current question.
    Returns evaluation and next question (or completion signal).
    """
    session_id = request.session_id

    if session_id not in _session_states:
        raise HTTPException(status_code=404, detail="Session not found")

    state = _session_states[session_id]
    graph = _session_graphs.get(session_id)

    if not graph:
        raise HTTPException(status_code=500, detail="Interview graph not found")

    if state.get("phase") == "complete":
        raise HTTPException(status_code=400, detail="Interview already completed")

    logger.info(f"Answer received for session: {session_id}")

    try:
        evaluation, next_question, updated_state = graph.submit_answer(
            state=state,
            answer=request.answer,
        )

        # Update session state
        _session_states[session_id] = updated_state

        # Persist to DB
        db = get_db()
        qa_item = {
            "question": state["current_question"],
            "answer": request.answer,
            "evaluation": evaluation,
            "is_follow_up": state.get("is_follow_up", False),
        }
        db.append_qa(session_id, qa_item)

        # Check if interview is complete
        is_complete = next_question == "__INTERVIEW_COMPLETE__"

        response_data = {
            "evaluation": evaluation,
            "score": evaluation.get("score", 0),
            "feedback": evaluation.get("feedback", ""),
            "is_complete": is_complete,
            "questions_answered": updated_state.get("question_count", 0),
            "total_questions": updated_state.get("max_questions", 8),
        }

        if not is_complete:
            q_num = updated_state.get("question_count", 0)
            is_followup = updated_state.get("is_follow_up", False)
            response_data.update({
                "next_question": next_question,
                "question_number": q_num + (1 if is_followup else 0),
                "is_follow_up": is_followup,
                "difficulty": updated_state.get("difficulty", "medium"),
            })

        return SessionResponse(
            session_id=session_id,
            message="Answer submitted" if not is_complete else "Interview complete",
            data=response_data,
        )

    except Exception as e:
        logger.error(f"Answer processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing answer: {str(e)}")


@app.post("/api/v1/interview/finalize", response_model=SessionResponse)
async def finalize_interview(session_id: str):
    """
    Generate the final HR decision report for a session.
    """
    if session_id not in _session_states:
        raise HTTPException(status_code=404, detail="Session not found")

    state = _session_states[session_id]
    graph = _session_graphs.get(session_id)

    if not graph:
        raise HTTPException(status_code=500, detail="Interview graph not found")

    if not state.get("qa_history"):
        raise HTTPException(status_code=400, detail="No Q&A history to evaluate")

    try:
        logger.info(f"Generating final report for session: {session_id}")
        report = graph.finalize_interview(state)

        # Update state
        state["final_report"] = report
        state["phase"] = "complete"
        _session_states[session_id] = state

        # Save to DB
        db = get_db()
        db.save_report(session_id, report)

        # Cleanup retriever
        clear_retriever(session_id)

        return SessionResponse(
            session_id=session_id,
            message="Interview finalized",
            data={"report": report}
        )

    except Exception as e:
        logger.error(f"Finalization error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@app.get("/api/v1/interview/status/{session_id}")
async def get_session_status(session_id: str):
    """Get current status of an interview session."""
    if session_id not in _session_states:
        # Try DB
        db = get_db()
        session = db.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session_id": session_id, "data": session}

    state = _session_states[session_id]
    from utils.scoring import calculate_average_score
    scores = state.get("scores", [])

    return {
        "session_id": session_id,
        "role": state.get("role"),
        "candidate_name": state.get("candidate_name"),
        "phase": state.get("phase"),
        "question_count": state.get("question_count", 0),
        "max_questions": state.get("max_questions", 8),
        "average_score": calculate_average_score(scores) if scores else None,
        "scores": scores,
        "is_complete": state.get("phase") == "complete",
        "current_question": state.get("current_question", ""),
        "has_resume": bool(state.get("resume_data")),
    }


@app.get("/api/v1/interview/report/{session_id}")
async def get_report(session_id: str):
    """Get the final report for a completed session."""
    # Check in-memory first
    if session_id in _session_states:
        state = _session_states[session_id]
        if state.get("final_report"):
            return {"session_id": session_id, "report": state["final_report"]}

    # Try DB
    db = get_db()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    report = session.get("final_report")
    if not report:
        raise HTTPException(status_code=404, detail="Report not yet generated")

    return {"session_id": session_id, "report": report}


@app.get("/api/v1/sessions")
async def list_sessions(limit: int = 20, status: Optional[str] = None):
    """List recent interview sessions."""
    db = get_db()
    sessions = db.list_sessions(limit=limit, status=status)
    return {"sessions": sessions, "count": len(sessions)}


@app.delete("/api/v1/interview/{session_id}")
async def delete_session(session_id: str):
    """Delete an interview session."""
    if session_id in _session_states:
        del _session_states[session_id]
    if session_id in _session_graphs:
        del _session_graphs[session_id]
    clear_retriever(session_id)

    db = get_db()
    deleted = db.delete_session(session_id)

    return {"message": "Session deleted" if deleted else "Session not found"}


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.APP_ENV == "development",
        log_level="info",
    )