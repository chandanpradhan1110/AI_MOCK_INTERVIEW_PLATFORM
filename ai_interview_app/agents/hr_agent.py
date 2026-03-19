"""
HR Decision Agent
Analyzes complete interview data and generates final hiring decision + report.
"""
import json
import re
from typing import Dict, Any, List, Optional
from loguru import logger


from utils.prompts import HR_DECISION_SYSTEM, HR_DECISION_PROMPT
from utils.scoring import (
    calculate_average_score,
    build_transcript,
    extract_scores_from_history,
    format_score_breakdown,
    score_to_grade,
)
from utils.llm_client import get_llm_client, get_model_name, supports_json_mode


class HRDecisionAgent:
    """
    Agent that makes the final hiring decision based on complete interview performance.
    
    Outputs:
    - Overall score
    - Hire / No Hire / Borderline decision
    - Strengths and weaknesses
    - Actionable recommendations
    """

    def __init__(self):
        self.client = get_llm_client()
        self.model = get_model_name()
        self.use_json_mode = supports_json_mode()
        logger.info("HRDecisionAgent initialized")

    def generate_report(
        self,
        role: str,
        qa_history: List[Dict[str, Any]],
        resume_summary: str = "No resume provided.",
    ) -> Dict[str, Any]:

        if not qa_history:
            return self._empty_interview_report()

        # Build inputs
        transcript = build_transcript(qa_history)
        scores = extract_scores_from_history(qa_history)
        avg_score = calculate_average_score(scores)
        score_str = ", ".join(f"Q{i+1}: {s}/10" for i, s in enumerate(scores))

        prompt = HR_DECISION_PROMPT.format(
            role=role,
            resume_summary=resume_summary[:500],
            transcript=transcript,
            scores=score_str,
            avg_score=round(avg_score, 2),
        )

        try:
            # ✅ Step 1: prepare kwargs
            kwargs = {}

            if self.use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            # ✅ Step 2: API call
            response = self.client.chat.completions.create(
                model=self.model,   # ✅ changed here
                messages=[
                    {"role": "system", "content": HR_DECISION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1200,
                **kwargs   # ✅ inject conditionally
            )

            content = response.choices[0].message.content.strip()

            # ✅ Step 3: parse JSON
            report = json.loads(content)

            # ✅ Step 4: enrich
            report = self._enrich_report(report, scores, avg_score, qa_history)

            logger.info(
                f"HR Report generated — Decision: {report.get('decision')} | "
                f"Score: {report.get('overall_score')}/10"
            )

            return report

        except json.JSONDecodeError as e:
            logger.error(f"HRDecisionAgent JSON parse error: {e}")
            return self._fallback_report(avg_score, scores)

        except Exception as e:
            logger.error(f"HRDecisionAgent error: {e}")
            return self._fallback_report(avg_score, scores)

    def _enrich_report(
        self,
        report: Dict[str, Any],
        scores: List[float],
        avg_score: float,
        qa_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Add computed fields to the LLM report."""
        score_breakdown = format_score_breakdown(qa_history)

        report["overall_score"] = round(avg_score, 2)
        report["grade"] = score_to_grade(avg_score)
        report["score_breakdown"] = score_breakdown
        report["total_questions"] = len(scores)
        report["individual_scores"] = scores

        # Ensure lists for strengths/weaknesses/recommendations
        for field in ["strengths", "weaknesses", "recommendations",
                      "technical_areas_strong", "technical_areas_weak"]:
            if field in report and isinstance(report[field], str):
                report[field] = [report[field]]
            elif field not in report:
                report[field] = []

        # Ensure decision is valid
        valid_decisions = {"Hire", "No Hire", "Borderline"}
        if report.get("decision") not in valid_decisions:
            report["decision"] = self._score_to_decision(avg_score)

        return report

    def _score_to_decision(self, avg_score: float) -> str:
        """Determine decision from score."""
        if avg_score >= 7.5:
            return "Hire"
        elif avg_score >= 5.5:
            return "Borderline"
        else:
            return "No Hire"

    def _empty_interview_report(self) -> Dict[str, Any]:
        """Report when no interview data is available."""
        return {
            "overall_score": 0,
            "grade": "N/A",
            "decision": "No Hire",
            "decision_reasoning": "No interview data available.",
            "strengths": [],
            "weaknesses": ["Did not complete the interview."],
            "recommendations": ["Complete the full interview to receive a proper evaluation."],
            "technical_areas_strong": [],
            "technical_areas_weak": [],
            "interview_summary": "The interview was not completed.",
            "total_questions": 0,
            "individual_scores": [],
        }

    def _fallback_report(self, avg_score: float, scores: List[float]) -> Dict[str, Any]:
        """Fallback report when LLM fails."""
        decision = self._score_to_decision(avg_score)
        return {
            "overall_score": round(avg_score, 2),
            "grade": score_to_grade(avg_score),
            "decision": decision,
            "decision_reasoning": f"Based on average score of {avg_score:.1f}/10.",
            "strengths": ["Completed the interview process."],
            "weaknesses": ["Detailed analysis unavailable."],
            "recommendations": [
                "Review areas where you scored below 6.",
                "Practice more complex problem-solving scenarios.",
            ],
            "technical_areas_strong": [],
            "technical_areas_weak": [],
            "interview_summary": (
                f"Candidate completed the interview with an average score of {avg_score:.1f}/10. "
                f"Decision: {decision}."
            ),
            "total_questions": len(scores),
            "individual_scores": scores,
        }

    def format_report_markdown(self, report: Dict[str, Any]) -> str:
        """Format the report as readable Markdown."""
        decision = report.get("decision", "N/A")
        decision_emoji = {"Hire": "✅", "No Hire": "❌", "Borderline": "⚠️"}.get(decision, "❓")

        lines = [
            f"# 📊 Interview Report",
            f"",
            f"## {decision_emoji} Decision: **{decision}**",
            f"**Overall Score:** {report.get('overall_score', 'N/A')}/10 (Grade: {report.get('grade', 'N/A')})",
            f"",
            f"### 📋 Summary",
            report.get("interview_summary", ""),
            f"",
            f"### 💡 Decision Reasoning",
            report.get("decision_reasoning", ""),
            f"",
            f"### ✅ Strengths",
        ]
        for s in report.get("strengths", []):
            lines.append(f"- {s}")

        lines += [f"", f"### ❌ Weaknesses"]
        for w in report.get("weaknesses", []):
            lines.append(f"- {w}")

        lines += [f"", f"### 📈 Recommendations"]
        for r in report.get("recommendations", []):
            lines.append(f"- {r}")

        lines += [
            f"",
            f"### 📊 Score Breakdown",
            f"Questions answered: {report.get('total_questions', 0)}",
            f"Scores: {', '.join(str(s) for s in report.get('individual_scores', []))}",
        ]

        return "\n".join(lines)