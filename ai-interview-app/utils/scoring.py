"""
Scoring utilities for the interview platform.
"""
from typing import List, Dict, Any
from loguru import logger


def calculate_average_score(scores: List[float]) -> float:
    """Calculate the average score from a list of scores."""
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 2)


def determine_difficulty(question_number: int, scores: List[float]) -> str:
    """
    Dynamically adjust difficulty based on performance so far.
    
    Args:
        question_number: Current question index (0-based)
        scores: List of scores so far

    Returns:
        Difficulty level string
    """
    if question_number == 0 or not scores:
        return "medium"

    avg = calculate_average_score(scores)

    if avg >= 8.0:
        return "hard"
    elif avg >= 6.0:
        return "medium-hard"
    elif avg >= 4.0:
        return "medium"
    else:
        return "easy"


def should_ask_followup(score: float, followup_count: int, max_followups: int = 2) -> bool:
    """
    Determine whether a follow-up question should be asked.
    
    Args:
        score: Score of the last answer (0-10)
        followup_count: Number of follow-ups already asked
        max_followups: Maximum follow-ups per session
        
    Returns:
        True if follow-up should be asked
    """
    if followup_count >= max_followups:
        return False
    # Always ask follow-up (nature differs by score)
    return True


def score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 9:
        return "A+"
    elif score >= 8:
        return "A"
    elif score >= 7:
        return "B+"
    elif score >= 6:
        return "B"
    elif score >= 5:
        return "C"
    elif score >= 4:
        return "D"
    else:
        return "F"


def build_transcript(qa_history: List[Dict[str, Any]]) -> str:
    """
    Build a formatted transcript from Q&A history.
    
    Args:
        qa_history: List of Q&A dicts
        
    Returns:
        Formatted transcript string
    """
    lines = []
    for i, item in enumerate(qa_history, 1):
        lines.append(f"Q{i}: {item.get('question', '')}")
        lines.append(f"A{i}: {item.get('answer', '')}")
        eval_data = item.get("evaluation", {})
        if eval_data:
            lines.append(f"Score: {eval_data.get('score', 'N/A')}/10")
            lines.append(f"Feedback: {eval_data.get('feedback', '')}")
        lines.append("---")
    return "\n".join(lines)


def extract_scores_from_history(qa_history: List[Dict[str, Any]]) -> List[float]:
    """Extract all numeric scores from Q&A history."""
    scores = []
    for item in qa_history:
        eval_data = item.get("evaluation", {})
        score = eval_data.get("score")
        if score is not None:
            try:
                scores.append(float(score))
            except (ValueError, TypeError):
                logger.warning(f"Could not parse score: {score}")
    return scores


def format_score_breakdown(qa_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a detailed score breakdown for the report.
    
    Returns:
        Dict with score statistics
    """
    scores = extract_scores_from_history(qa_history)
    if not scores:
        return {"error": "No scores available"}

    return {
        "scores": scores,
        "average": calculate_average_score(scores),
        "highest": max(scores),
        "lowest": min(scores),
        "grade": score_to_grade(calculate_average_score(scores)),
        "total_questions": len(scores),
        "questions_above_7": sum(1 for s in scores if s >= 7),
        "questions_below_5": sum(1 for s in scores if s < 5),
    }