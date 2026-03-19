"""
Candidate Evaluator Agent
Evaluates candidate answers with strict JSON output.
"""
import json
import re
from typing import Dict, Any, Optional
from loguru import logger


from utils.prompts import EVALUATOR_SYSTEM, EVALUATOR_PROMPT
from utils.llm_client import get_llm_client, get_model_name, supports_json_mode


class EvaluatorAgent:
    """
    Agent that evaluates candidate answers in real-time.
    
    Returns structured JSON with score, feedback, strengths, and improvements.
    """

    def __init__(self):
        self.client = get_llm_client()
        self.model = get_model_name()
        self.use_json_mode = supports_json_mode()
        logger.info("EvaluatorAgent initialized")

    def evaluate(
        self,
        question: str,
        answer: str,
        role: str,
    ) -> Dict[str, Any]:

        if not answer or len(answer.strip()) < 5:
            return self._empty_answer_evaluation()

        prompt = EVALUATOR_PROMPT.format(
            question=question,
            answer=answer,
            role=role,
        )

        try:
            # ✅ Step 1: prepare kwargs
            kwargs = {}

            if self.use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            # ✅ Step 2: API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EVALUATOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=600,
                **kwargs   # ✅ inject here
            )

            content = response.choices[0].message.content.strip()

            # ✅ Step 3: parse JSON
            evaluation = json.loads(content)

            # ✅ Step 4: validate
            evaluation = self._validate_evaluation(evaluation)

            logger.info(
                f"Evaluated answer — Score: {evaluation['score']}/10 | "
                f"Q: {question[:50]}..."
            )

            return evaluation

        except json.JSONDecodeError as e:
            logger.error(f"EvaluatorAgent JSON parse error: {e}")
            return self._parse_from_text(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"EvaluatorAgent error: {e}")
            return self._fallback_evaluation(answer)

    def _validate_evaluation(self, eval_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure the evaluation has all required fields and valid types."""
        # Ensure score is in range
        score = eval_dict.get("score", 5)
        try:
            score = float(score)
        except (ValueError, TypeError):
            score = 5.0
        score = max(0.0, min(10.0, score))

        return {
            "score": score,
            "feedback": str(eval_dict.get("feedback", "No feedback provided.")),
            "strengths": str(eval_dict.get("strengths", "Answer provided.")),
            "improvement": str(eval_dict.get("improvement", "Provide more detail.")),
            "technical_accuracy": str(eval_dict.get("technical_accuracy", "Not assessed.")),
        }

    def _parse_from_text(self, text: str) -> Dict[str, Any]:
        """Attempt to extract score from malformed response."""
        score = 5.0
        score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text)
        if score_match:
            score = float(score_match.group(1))

        return {
            "score": score,
            "feedback": "Answer was evaluated. Please see the score.",
            "strengths": "Response provided.",
            "improvement": "Consider providing more detail and examples.",
            "technical_accuracy": "Partially assessed.",
        }

    def _empty_answer_evaluation(self) -> Dict[str, Any]:
        """Return evaluation for empty/very short answers."""
        return {
            "score": 0.0,
            "feedback": "No answer was provided. A complete response is required.",
            "strengths": "N/A",
            "improvement": "Please provide a detailed answer addressing the question.",
            "technical_accuracy": "Cannot assess — no answer given.",
        }

    def _fallback_evaluation(self, answer: str) -> Dict[str, Any]:
        """Simple fallback evaluation when LLM fails."""
        word_count = len(answer.split())
        # Basic heuristic scoring
        if word_count < 10:
            score = 2.0
        elif word_count < 30:
            score = 4.0
        elif word_count < 80:
            score = 5.5
        else:
            score = 6.0

        return {
            "score": score,
            "feedback": "Your answer was received. Evaluation service temporarily unavailable.",
            "strengths": "You provided a response.",
            "improvement": "Aim to provide detailed, structured answers with examples.",
            "technical_accuracy": "Unable to assess at this time.",
        }

    def get_live_feedback_summary(self, evaluation: Dict[str, Any]) -> str:
        """Format evaluation for real-time display."""
        score = evaluation.get("score", 0)
        emoji = "🟢" if score >= 7 else "🟡" if score >= 5 else "🔴"
        return (
            f"{emoji} Score: **{score}/10**\n\n"
            f"📝 **Feedback:** {evaluation.get('feedback', '')}\n\n"
            f"✅ **Strengths:** {evaluation.get('strengths', '')}\n\n"
            f"📈 **To Improve:** {evaluation.get('improvement', '')}"
        )