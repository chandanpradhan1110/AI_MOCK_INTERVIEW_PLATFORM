"""
Follow-up Agent
Generates contextual follow-up questions based on previous answer quality.
- score < 6 → basic/fundamental follow-up
- score >= 6 → advanced/system-design follow-up
"""
from typing import Optional
from loguru import logger
from openai import OpenAI

from utils.prompts import (
    FOLLOWUP_SYSTEM,
    FOLLOWUP_PROMPT_LOW,
    FOLLOWUP_PROMPT_HIGH,
)
from config import settings


class FollowUpAgent:
    """
    Agent that generates intelligent follow-up questions.
    Adapts difficulty based on the candidate's score.
    """

    THRESHOLD = 6.0

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("FollowUpAgent initialized")

    def generate_followup(
        self,
        previous_question: str,
        answer: str,
        score: float,
    ) -> str:
        """
        Generate a follow-up question based on score.

        Args:
            previous_question: The question that was asked
            answer: The candidate's answer
            score: Score received (0-10)

        Returns:
            Follow-up question string
        """
        is_high_score = score >= self.THRESHOLD

        if is_high_score:
            prompt = FOLLOWUP_PROMPT_HIGH.format(
                previous_question=previous_question,
                answer=answer,
                score=score,
            )
            logger.info(f"Generating ADVANCED follow-up (score={score})")
        else:
            prompt = FOLLOWUP_PROMPT_LOW.format(
                previous_question=previous_question,
                answer=answer,
                score=score,
            )
            logger.info(f"Generating BASIC follow-up (score={score})")

        try:
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": FOLLOWUP_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.75,
                max_tokens=250,
            )
            followup = response.choices[0].message.content.strip()

            # Clean up any preamble the model might add
            followup = self._clean_question(followup)
            logger.info(f"Follow-up generated: {followup[:80]}...")
            return followup

        except Exception as e:
            logger.error(f"FollowUpAgent error: {e}")
            return self._fallback_followup(previous_question, is_high_score)

    def _clean_question(self, text: str) -> str:
        """Remove any preamble from the generated question."""
        # Remove common LLM preambles
        prefixes_to_remove = [
            "Follow-up question:",
            "Follow-up:",
            "Here's a follow-up:",
            "Sure,",
            "Great,",
            "Certainly,",
        ]
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Ensure ends with ?
        if text and not text.endswith("?"):
            text = text + "?"

        return text

    def _fallback_followup(self, previous_question: str, advanced: bool) -> str:
        """Fallback follow-up questions."""
        if advanced:
            return (
                "Interesting! Now let's think about scale — "
                "how would your approach change if you had to handle this "
                "with 100x more data or traffic?"
            )
        else:
            return (
                "Let me rephrase that slightly — "
                "can you explain the core concept behind your answer "
                "in the simplest terms possible?"
            )

    def get_followup_type(self, score: float) -> str:
        """Return the type of follow-up that will be asked."""
        return "advanced" if score >= self.THRESHOLD else "fundamental"