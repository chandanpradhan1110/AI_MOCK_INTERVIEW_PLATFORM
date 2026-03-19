"""
Question Generator Agent
Generates personalized interview questions based on role, resume, and JD context.
"""
from typing import List, Optional, Dict, Any
from loguru import logger


from utils.prompts import QUESTION_GENERATOR_SYSTEM, QUESTION_GENERATOR_PROMPT
from utils.llm_client import get_llm_client, get_model_name


class QuestionGeneratorAgent:
    """
    Agent responsible for generating targeted interview questions.
    
    Incorporates:
    - Role-based question banks
    - Resume-specific personalization
    - JD-based RAG context
    - Dynamic difficulty adjustment
    """

    def __init__(self):
        self.client = get_llm_client()
        self.model = get_model_name()
        logger.info("QuestionGeneratorAgent initialized")

    def generate_question(
        self,
        role: str,
        question_number: int,
        total_questions: int,
        difficulty: str = "medium",
        resume_context: str = "No resume provided.",
        jd_context: str = "No job description provided.",
        previous_questions: Optional[List[str]] = None,
    ) -> str:
        """
        Generate the next interview question.

        Args:
            role: Target role (e.g., "Data Scientist")
            question_number: Current question index (1-based)
            total_questions: Total questions planned
            difficulty: Difficulty level
            resume_context: Structured resume context string
            jd_context: Retrieved JD context string
            previous_questions: List of already-asked questions

        Returns:
            Generated question string
        """
        prev_q_str = (
            "\n".join(f"- {q}" for q in previous_questions)
            if previous_questions else "None"
        )

        prompt = QUESTION_GENERATOR_PROMPT.format(
            role=role,
            difficulty=difficulty,
            resume_context=resume_context,
            jd_context=jd_context,
            previous_questions=prev_q_str,
            question_number=question_number,
            total_questions=total_questions,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QUESTION_GENERATOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,  # Higher for variety
                max_tokens=300,
            )
            question = response.choices[0].message.content.strip()
            logger.info(f"Generated Q{question_number}: {question[:80]}...")
            return question

        except Exception as e:
            logger.error(f"QuestionGeneratorAgent error: {e}")
            # Fallback to role-based generic question
            return self._fallback_question(role, difficulty, question_number)

    def _fallback_question(self, role: str, difficulty: str, q_num: int) -> str:
        """Generate a fallback question from predefined pools."""
        fallback_pools = {
            "Data Scientist": [
                "Explain the bias-variance tradeoff and how you handle it in practice.",
                "Walk me through how you would approach a classification problem end-to-end.",
                "How do you handle missing data in a real-world dataset?",
                "Explain cross-validation and why it matters.",
                "What's the difference between bagging and boosting?",
                "How do you detect and handle data leakage?",
                "Explain gradient descent and its variants.",
            ],
            "ML Engineer": [
                "How do you deploy a machine learning model to production?",
                "Explain the concept of model drift and how you monitor for it.",
                "What's your approach to feature engineering for tabular data?",
                "How do you optimize model inference latency?",
                "Describe your experience with MLOps pipelines.",
                "What's the difference between batch and online learning?",
            ],
            "Python Developer": [
                "Explain Python's GIL and its implications for concurrency.",
                "How would you design a RESTful API in Python?",
                "What are Python decorators and how do you use them?",
                "Explain Python's memory management and garbage collection.",
                "How do you handle async programming in Python?",
            ],
        }

        pool = fallback_pools.get(role, fallback_pools["Python Developer"])
        return pool[(q_num - 1) % len(pool)]

    def generate_opening_question(
        self,
        role: str,
        candidate_name: str = "the candidate",
        resume_context: str = "",
    ) -> str:
        """Generate a warm opening question to start the interview."""
        if resume_context and resume_context != "No resume provided.":
            opening = (
                f"Welcome! I've reviewed your background. "
                f"Let's start — can you give me a brief overview of your experience "
                f"and what specifically draws you to the {role} role?"
            )
        else:
            opening = (
                f"Welcome to your {role} interview! "
                f"Let's start with a brief introduction — can you walk me through "
                f"your background and what you've been working on recently?"
            )
        return opening