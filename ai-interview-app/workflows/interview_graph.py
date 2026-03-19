"""
LangGraph Interview Workflow
Orchestrates all agents in a stateful multi-agent graph.

Graph Flow:
START → initialize → generate_question → [wait_for_answer] → evaluate
      → follow_up (conditional) → [wait_for_followup_answer] → evaluate
      → check_completion → (loop OR) generate_hr_report → END
"""
import json
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from loguru import logger

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

from agents.question_agent import QuestionGeneratorAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.followup_agent import FollowUpAgent
from agents.hr_agent import HRDecisionAgent
from rag.retriever import FAISSRetriever
from utils.scoring import (
    determine_difficulty,
    extract_scores_from_history,
    calculate_average_score,
)
from config import settings


# ─────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────

class InterviewState(TypedDict):
    """Complete state of the interview session."""
    # Session info
    session_id: str
    role: str
    candidate_name: str

    # Context
    resume_context: str
    resume_data: Dict[str, Any]
    jd_context: str

    # Interview progress
    qa_history: List[Dict[str, Any]]
    current_question: str
    current_answer: str
    current_evaluation: Dict[str, Any]
    previous_questions: List[str]

    # Follow-up tracking
    follow_up_count: int
    max_follow_ups: int
    is_follow_up: bool

    # Scoring
    scores: List[float]
    difficulty: str

    # Control
    question_count: int
    max_questions: int
    phase: str  # "questioning" | "followup" | "complete"
    waiting_for_answer: bool

    # Final output
    final_report: Optional[Dict[str, Any]]
    error: Optional[str]


# ─────────────────────────────────────────────
# Interview Graph Builder
# ─────────────────────────────────────────────

class InterviewGraph:
    """
    LangGraph-based multi-agent interview orchestrator.
    Manages the full interview lifecycle.
    """

    def __init__(self, retriever: Optional[FAISSRetriever] = None):
        self.question_agent = QuestionGeneratorAgent()
        self.evaluator_agent = EvaluatorAgent()
        self.followup_agent = FollowUpAgent()
        self.hr_agent = HRDecisionAgent()
        self.retriever = retriever
        self.graph = self._build_graph()
        logger.info("InterviewGraph initialized")

    def _build_graph(self) -> Any:
        """Build the LangGraph state machine."""
        workflow = StateGraph(InterviewState)

        # Add nodes
        workflow.add_node("generate_question", self._generate_question_node)
        workflow.add_node("evaluate_answer", self._evaluate_answer_node)
        workflow.add_node("generate_followup", self._generate_followup_node)
        workflow.add_node("generate_hr_report", self._generate_hr_report_node)

        # Entry point
        workflow.set_entry_point("generate_question")

        # Edges
        workflow.add_edge("generate_question", END)  # Pause for user input
        workflow.add_edge("evaluate_answer", "generate_followup")  # Always check for followup
        workflow.add_conditional_edges(
            "generate_followup",
            self._route_after_followup,
            {
                "continue": "generate_question",
                "complete": "generate_hr_report",
                "followup_answer": END,  # Pause for follow-up answer
            }
        )
        workflow.add_edge("generate_hr_report", END)

        return workflow.compile()

    # ─────────────────────────────────────────
    # Node Implementations
    # ─────────────────────────────────────────

    def _generate_question_node(self, state: InterviewState) -> Dict[str, Any]:
        """Generate the next interview question."""
        # Check if interview is done
        if state["question_count"] >= state["max_questions"]:
            return {"phase": "complete"}

        # Determine difficulty dynamically
        difficulty = determine_difficulty(state["question_count"], state["scores"])

        # Get JD context from retriever if available
        jd_context = state.get("jd_context", "")
        if self.retriever and self.retriever.is_ready():
            # Use role as query to get relevant JD sections
            query = f"{state['role']} technical skills requirements"
            jd_context = self.retriever.retrieve_context_string(query, top_k=3)

        question = self.question_agent.generate_question(
            role=state["role"],
            question_number=state["question_count"] + 1,
            total_questions=state["max_questions"],
            difficulty=difficulty,
            resume_context=state.get("resume_context", ""),
            jd_context=jd_context,
            previous_questions=state.get("previous_questions", []),
        )

        logger.info(f"[Q{state['question_count']+1}] {question[:80]}...")

        return {
            "current_question": question,
            "current_answer": "",
            "current_evaluation": {},
            "is_follow_up": False,
            "difficulty": difficulty,
            "jd_context": jd_context,
            "waiting_for_answer": True,
            "phase": "questioning",
        }

    def _evaluate_answer_node(self, state: InterviewState) -> Dict[str, Any]:
        """Evaluate the candidate's answer."""
        evaluation = self.evaluator_agent.evaluate(
            question=state["current_question"],
            answer=state["current_answer"],
            role=state["role"],
        )

        score = evaluation.get("score", 0)
        new_scores = state["scores"] + [score]

        # Build Q&A record
        qa_item = {
            "question_number": state["question_count"] + 1,
            "question": state["current_question"],
            "answer": state["current_answer"],
            "evaluation": evaluation,
            "is_follow_up": state.get("is_follow_up", False),
            "difficulty": state.get("difficulty", "medium"),
        }

        updated_history = state["qa_history"] + [qa_item]
        updated_prev_questions = state["previous_questions"] + [state["current_question"]]

        logger.info(f"Answer evaluated — Score: {score}/10")

        return {
            "scores": new_scores,
            "current_evaluation": evaluation,
            "qa_history": updated_history,
            "previous_questions": updated_prev_questions,
            "question_count": state["question_count"] + 1,
            "waiting_for_answer": False,
        }

    def _generate_followup_node(self, state: InterviewState) -> Dict[str, Any]:
        """Decide and generate follow-up question."""
        # Check if we should ask a follow-up
        follow_up_count = state.get("follow_up_count", 0)
        max_follow_ups = state.get("max_follow_ups", settings.FOLLOW_UP_QUESTIONS)
        last_score = state["scores"][-1] if state["scores"] else 5.0

        should_followup = (
            follow_up_count < max_follow_ups
            and state["question_count"] < state["max_questions"]
        )

        if not should_followup:
            # No follow-up — either continue to next main question or complete
            return {
                "is_follow_up": False,
                "phase": "complete" if state["question_count"] >= state["max_questions"] else "questioning",
            }

        # Generate follow-up
        followup_q = self.followup_agent.generate_followup(
            previous_question=state["current_question"],
            answer=state["current_answer"],
            score=last_score,
        )

        followup_type = self.followup_agent.get_followup_type(last_score)
        logger.info(f"Follow-up type: {followup_type} | Q: {followup_q[:60]}...")

        return {
            "current_question": followup_q,
            "current_answer": "",
            "is_follow_up": True,
            "follow_up_count": follow_up_count + 1,
            "waiting_for_answer": True,
            "phase": "followup",
        }

    def _generate_hr_report_node(self, state: InterviewState) -> Dict[str, Any]:
        """Generate the final HR decision report."""
        logger.info("Generating final HR report...")

        resume_summary = state.get("resume_context", "No resume provided.")[:300]

        report = self.hr_agent.generate_report(
            role=state["role"],
            qa_history=state["qa_history"],
            resume_summary=resume_summary,
        )

        return {
            "final_report": report,
            "phase": "complete",
        }

    # ─────────────────────────────────────────
    # Routing Logic
    # ─────────────────────────────────────────

    def _route_after_followup(self, state: InterviewState) -> str:
        """Route after follow-up node decision."""
        if state.get("phase") == "complete":
            return "complete"
        elif state.get("is_follow_up") and state.get("waiting_for_answer"):
            return "followup_answer"  # Pause for user answer to follow-up
        elif state["question_count"] >= state["max_questions"]:
            return "complete"
        else:
            return "continue"

    # ─────────────────────────────────────────
    # Public API (Session-based)
    # ─────────────────────────────────────────

    def initialize_state(
        self,
        session_id: str,
        role: str,
        candidate_name: str = "Candidate",
        resume_context: str = "",
        resume_data: Optional[Dict] = None,
        max_questions: int = None,
    ) -> InterviewState:
        """Create the initial state for a new interview."""
        return InterviewState(
            session_id=session_id,
            role=role,
            candidate_name=candidate_name,
            resume_context=resume_context or "No resume provided.",
            resume_data=resume_data or {},
            jd_context="No job description provided.",
            qa_history=[],
            current_question="",
            current_answer="",
            current_evaluation={},
            previous_questions=[],
            follow_up_count=0,
            max_follow_ups=settings.FOLLOW_UP_QUESTIONS,
            is_follow_up=False,
            scores=[],
            difficulty="medium",
            question_count=0,
            max_questions=max_questions or settings.MAX_QUESTIONS,
            phase="questioning",
            waiting_for_answer=False,
            final_report=None,
            error=None,
        )

    def get_next_question(self, state: InterviewState) -> tuple[str, InterviewState]:
        """
        Get the next question for the interview.
        
        Returns:
            (question_text, updated_state)
        """
        result = self.graph.invoke(state, config={"recursion_limit": 5})
        updated_state = {**state, **result}
        question = updated_state.get("current_question", "")
        return question, updated_state

    def submit_answer(
        self, state: InterviewState, answer: str
    ) -> tuple[Dict[str, Any], str, InterviewState]:
        """
        Submit an answer and get evaluation + next question.
        
        Returns:
            (evaluation, next_question, updated_state)
        """
        # Set the answer
        state_with_answer = {**state, "current_answer": answer}

        # Run evaluation node
        eval_result = self._evaluate_answer_node(state_with_answer)
        state_after_eval = {**state_with_answer, **eval_result}

        # Run follow-up node
        followup_result = self._generate_followup_node(state_after_eval)
        state_after_followup = {**state_after_eval, **followup_result}

        evaluation = state_after_eval.get("current_evaluation", {})

        # Check if we have a follow-up to present
        if state_after_followup.get("is_follow_up") and state_after_followup.get("waiting_for_answer"):
            next_question = state_after_followup.get("current_question", "")
            return evaluation, next_question, state_after_followup

        # Check if interview is complete
        if (state_after_followup.get("question_count", 0) >= state_after_followup.get("max_questions", 8)
                or state_after_followup.get("phase") == "complete"):
            return evaluation, "__INTERVIEW_COMPLETE__", state_after_followup

        # Generate next main question
        next_q_result = self._generate_question_node(state_after_followup)
        final_state = {**state_after_followup, **next_q_result}
        next_question = final_state.get("current_question", "")

        return evaluation, next_question, final_state

    def finalize_interview(self, state: InterviewState) -> Dict[str, Any]:
        """Generate and return the final HR report."""
        report_result = self._generate_hr_report_node(state)
        return report_result.get("final_report", {})