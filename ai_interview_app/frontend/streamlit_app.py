"""
Streamlit Frontend for AI Mock Interview Platform
Full-featured interview UI with real-time feedback.
"""
import streamlit as st
import requests
import json
import time
from typing import Optional, Dict, Any

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

API_BASE_URL = "http://localhost:8000/api/v1"

ROLES = [
    "Data Scientist",
    "ML Engineer",
    "Python Developer",
    "Data Analyst",
    "Backend Engineer",
    "AI Research Engineer",
]

ROLE_DESCRIPTIONS = {
    "Data Scientist": "🔬 Statistics, ML modeling, feature engineering, experimentation",
    "ML Engineer": "⚙️ Model deployment, MLOps, production pipelines, scaling",
    "Python Developer": "🐍 OOP, APIs, async, testing, architecture patterns",
    "Data Analyst": "📊 SQL, data visualization, business insights, reporting",
    "Backend Engineer": "🏗️ System design, databases, APIs, distributed systems",
    "AI Research Engineer": "🧠 Deep learning, transformers, research implementation",
}


# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI Mock Interview Platform",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .question-box {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    .evaluation-box {
        background: #f0fff4;
        border: 1px solid #68d391;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .score-badge-high { 
        background: #48bb78; color: white; 
        padding: 0.3rem 1rem; border-radius: 20px; font-weight: bold; 
    }
    .score-badge-mid { 
        background: #ed8936; color: white; 
        padding: 0.3rem 1rem; border-radius: 20px; font-weight: bold; 
    }
    .score-badge-low { 
        background: #fc8181; color: white; 
        padding: 0.3rem 1rem; border-radius: 20px; font-weight: bold; 
    }
    .hire-badge { 
        background: #48bb78; color: white; 
        padding: 0.5rem 2rem; border-radius: 25px; font-size: 1.3rem; font-weight: bold; 
    }
    .nohire-badge { 
        background: #fc8181; color: white; 
        padding: 0.5rem 2rem; border-radius: 25px; font-size: 1.3rem; font-weight: bold; 
    }
    .borderline-badge { 
        background: #ed8936; color: white; 
        padding: 0.5rem 2rem; border-radius: 25px; font-size: 1.3rem; font-weight: bold; 
    }
    .sidebar-info {
        background: #edf2ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .progress-section {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────

def init_session_state():
    defaults = {
        "page": "home",           # home | interview | report
        "session_id": None,
        "current_question": "",
        "question_number": 1,
        "total_questions": 8,
        "is_follow_up": False,
        "qa_history": [],         # Local display history
        "scores": [],
        "role": "",
        "candidate_name": "",
        "interview_started": False,
        "interview_complete": False,
        "final_report": None,
        "last_evaluation": None,
        "show_evaluation": False,
        "answer_submitted": False,
        "has_resume": False,
        "has_jd": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ─────────────────────────────────────────────
# API Helpers
# ─────────────────────────────────────────────

def api_start_interview(role: str, candidate_name: str, jd_text: str = "", max_q: int = 8) -> Optional[Dict]:
    try:
        payload = {
            "role": role,
            "candidate_name": candidate_name,
            "job_description": jd_text if jd_text else None,
            "max_questions": max_q,
        }
        resp = requests.post(f"{API_BASE_URL}/interview/start", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API. Make sure the server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"❌ Error starting interview: {e}")
        return None


def api_start_with_resume(
    role: str, candidate_name: str, resume_bytes: bytes = None,
    jd_text: str = "", max_q: int = 8
) -> Optional[Dict]:
    try:
        files = {}
        data = {
            "role": role,
            "candidate_name": candidate_name,
            "max_questions": str(max_q),
        }
        if jd_text:
            data["job_description"] = jd_text
        if resume_bytes:
            files["resume"] = ("resume.pdf", resume_bytes, "application/pdf")

        resp = requests.post(
            f"{API_BASE_URL}/interview/start-with-resume",
            data=data,
            files=files if files else None,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API. Make sure the server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None


def api_submit_answer(session_id: str, answer: str) -> Optional[Dict]:
    try:
        payload = {"session_id": session_id, "answer": answer}
        resp = requests.post(f"{API_BASE_URL}/interview/answer", json=payload, timeout=45)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"❌ Error submitting answer: {e}")
        return None


def api_finalize(session_id: str) -> Optional[Dict]:
    try:
        resp = requests.post(
            f"{API_BASE_URL}/interview/finalize",
            params={"session_id": session_id},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"❌ Error generating report: {e}")
        return None


# ─────────────────────────────────────────────
# UI Components
# ─────────────────────────────────────────────

def render_score_badge(score: float) -> str:
    if score >= 7:
        return f'<span class="score-badge-high">⭐ {score}/10</span>'
    elif score >= 5:
        return f'<span class="score-badge-mid">🟡 {score}/10</span>'
    else:
        return f'<span class="score-badge-low">⚠️ {score}/10</span>'


def render_sidebar():
    """Render sidebar with session info and controls."""
    with st.sidebar:
        st.markdown("## 🎤 AI Interview")
        st.markdown("---")

        if st.session_state.interview_started:
            # Session info
            st.markdown(f"**👤 Candidate:** {st.session_state.candidate_name}")
            st.markdown(f"**🎯 Role:** {st.session_state.role}")

            if st.session_state.has_resume:
                st.success("📄 Resume: Uploaded")
            if st.session_state.has_jd:
                st.success("📋 JD: Indexed")

            st.markdown("---")

            # Progress
            answered = len(st.session_state.scores)
            total = st.session_state.total_questions
            progress = min(answered / max(total, 1), 1.0)
            st.markdown(f"**Progress:** {answered}/{total} questions")
            st.progress(progress)

            # Score trend
            if st.session_state.scores:
                avg = sum(st.session_state.scores) / len(st.session_state.scores)
                st.markdown(f"**Average Score:** {avg:.1f}/10")

                # Score history
                st.markdown("**Score History:**")
                for i, score in enumerate(st.session_state.scores, 1):
                    bar = "🟢" if score >= 7 else "🟡" if score >= 5 else "🔴"
                    st.markdown(f"Q{i}: {bar} {score}/10")

            st.markdown("---")
            if st.button("🚪 End Interview", use_container_width=True):
                if st.session_state.session_id and st.session_state.scores:
                    with st.spinner("Generating report..."):
                        result = api_finalize(st.session_state.session_id)
                        if result:
                            st.session_state.final_report = result["data"]["report"]
                            st.session_state.page = "report"
                            st.rerun()

        else:
            st.markdown("""
            <div class="sidebar-info">
            <b>How it works:</b><br>
            1. Select your target role<br>
            2. Upload resume (optional)<br>
            3. Paste job description (optional)<br>
            4. Answer questions in real-time<br>
            5. Get AI-powered feedback + report
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Features:**")
            st.markdown("✅ Resume-personalized questions")
            st.markdown("✅ JD-based RAG questions")
            st.markdown("✅ Adaptive difficulty")
            st.markdown("✅ Real-time evaluation")
            st.markdown("✅ Final hiring decision")


# ─────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────

def page_home():
    """Home / Setup page."""
    st.markdown("""
    <div class="main-header">
        <h1>🎤 AI Mock Interview Platform</h1>
        <p>Personalized • Adaptive • Real-time Feedback</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 🎯 Interview Setup")

        # Role selection
        selected_role = st.selectbox(
            "Target Role",
            ROLES,
            help="Choose the role you're interviewing for"
        )
        if selected_role:
            st.caption(ROLE_DESCRIPTIONS.get(selected_role, ""))

        candidate_name = st.text_input(
            "Your Name",
            placeholder="Enter your name",
            value="Candidate"
        )

        # Number of questions
        max_questions = st.slider(
            "Number of Questions",
            min_value=3,
            max_value=15,
            value=8,
            help="Total main questions (follow-ups are additional)"
        )

    with col2:
        st.markdown("### 📂 Upload Documents")

        # Resume upload
        resume_file = st.file_uploader(
            "📄 Upload Resume (PDF)",
            type=["pdf"],
            help="Upload your resume for personalized questions"
        )

        # JD input
        jd_tab1, jd_tab2 = st.tabs(["📝 Paste JD Text", "📎 Upload JD PDF"])

        with jd_tab1:
            jd_text = st.text_area(
                "Job Description",
                placeholder="Paste the job description here...\nThe AI will tailor questions to match the JD requirements.",
                height=180,
            )

        with jd_tab2:
            jd_file = st.file_uploader(
                "Upload JD PDF",
                type=["pdf"],
                key="jd_pdf"
            )
            if jd_file:
                st.success(f"✅ JD uploaded: {jd_file.name}")

    st.markdown("---")

    # Start button
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        if st.button("🚀 Start Interview", use_container_width=True, type="primary"):
            if not selected_role:
                st.error("Please select a role")
                return

            with st.spinner("🤖 Initializing your personalized interview..."):
                # Determine if using resume/JD
                resume_bytes = resume_file.read() if resume_file else None

                # Merge JD text from file if uploaded
                final_jd = jd_text
                if jd_file and not jd_text:
                    # Read JD file (just pass bytes to API)
                    pass

                if resume_bytes or (jd_file and not jd_text):
                    result = api_start_with_resume(
                        role=selected_role,
                        candidate_name=candidate_name,
                        resume_bytes=resume_bytes,
                        jd_text=final_jd,
                        max_q=max_questions,
                    )
                else:
                    result = api_start_interview(
                        role=selected_role,
                        candidate_name=candidate_name,
                        jd_text=final_jd,
                        max_q=max_questions,
                    )

                if result:
                    data = result.get("data", {})
                    st.session_state.session_id = result["session_id"]
                    st.session_state.current_question = data.get("question", "")
                    st.session_state.question_number = 1
                    st.session_state.total_questions = data.get("total_questions", max_questions)
                    st.session_state.role = selected_role
                    st.session_state.candidate_name = data.get("candidate_name", candidate_name)
                    st.session_state.interview_started = True
                    st.session_state.has_resume = data.get("has_resume", bool(resume_bytes))
                    st.session_state.has_jd = data.get("has_jd", bool(final_jd))
                    st.session_state.page = "interview"
                    st.rerun()


def page_interview():
    """Main interview page."""
    if not st.session_state.interview_started:
        st.session_state.page = "home"
        st.rerun()

    # Header
    st.markdown(f"### 🎤 {st.session_state.role} Interview")

    # Progress bar
    answered = len(st.session_state.scores)
    total = st.session_state.total_questions
    progress = min(answered / max(total, 1), 1.0)

    col_prog, col_score = st.columns([3, 1])
    with col_prog:
        st.progress(progress, text=f"Question {answered + 1} of {total}")
    with col_score:
        if st.session_state.scores:
            avg = sum(st.session_state.scores) / len(st.session_state.scores)
            st.metric("Avg Score", f"{avg:.1f}/10")

    st.markdown("---")

    # Show previous evaluation if any
    if st.session_state.show_evaluation and st.session_state.last_evaluation:
        eval_data = st.session_state.last_evaluation
        score = eval_data.get("score", 0)

        with st.expander("📊 Previous Answer Evaluation", expanded=True):
            col_ev1, col_ev2 = st.columns([1, 3])
            with col_ev1:
                st.markdown(render_score_badge(score), unsafe_allow_html=True)
            with col_ev2:
                st.markdown(f"**Feedback:** {eval_data.get('feedback', '')}")

            col_s, col_i = st.columns(2)
            with col_s:
                st.success(f"✅ **Strengths:** {eval_data.get('strengths', '')}")
            with col_i:
                st.warning(f"📈 **To Improve:** {eval_data.get('improvement', '')}")

    # Current question
    is_followup = st.session_state.is_follow_up
    question_label = "🔄 Follow-up Question" if is_followup else f"❓ Question {st.session_state.question_number}"

    st.markdown(f"**{question_label}**")
    st.markdown(f"""
    <div class="question-box">
        {st.session_state.current_question}
    </div>
    """, unsafe_allow_html=True)

    # Tips
    if not is_followup:
        with st.expander("💡 Interview Tips", expanded=False):
            st.markdown("""
            - **Structure your answer** using the STAR method (Situation, Task, Action, Result)
            - **Be specific** — mention technologies, metrics, and concrete examples
            - **Show depth** — don't just define concepts, explain how you apply them
            - **Think out loud** — interviewers appreciate your reasoning process
            """)

    # Answer input
    answer = st.text_area(
        "Your Answer",
        placeholder="Type your detailed answer here...\n\nTip: Be specific and use real-world examples.",
        height=180,
        key=f"answer_{st.session_state.question_number}_{is_followup}",
    )

    col_sub, col_skip = st.columns([3, 1])
    with col_sub:
        submit_clicked = st.button(
            "📤 Submit Answer",
            use_container_width=True,
            type="primary",
            disabled=not answer.strip(),
        )
    with col_skip:
        skip_clicked = st.button("⏭️ Skip", use_container_width=True)

    if skip_clicked:
        answer = "I'd like to skip this question."
        submit_clicked = True

    if submit_clicked and answer.strip():
        with st.spinner("🤖 Evaluating your answer..."):
            result = api_submit_answer(st.session_state.session_id, answer)

            if result:
                data = result.get("data", {})
                evaluation = data.get("evaluation", {})
                score = evaluation.get("score", 0)

                # Update state
                st.session_state.scores.append(score)
                st.session_state.last_evaluation = evaluation
                st.session_state.show_evaluation = True

                # Add to local history
                st.session_state.qa_history.append({
                    "question": st.session_state.current_question,
                    "answer": answer,
                    "score": score,
                    "is_follow_up": is_followup,
                })

                if data.get("is_complete"):
                    # Generate final report
                    with st.spinner("📊 Generating your interview report..."):
                        report_result = api_finalize(st.session_state.session_id)
                        if report_result:
                            st.session_state.final_report = report_result["data"]["report"]
                        else:
                            # Use any partial report data
                            st.session_state.final_report = {"error": "Report generation failed"}
                    st.session_state.page = "report"
                    st.rerun()
                else:
                    # Continue interview
                    st.session_state.current_question = data.get("next_question", "")
                    st.session_state.is_follow_up = data.get("is_follow_up", False)
                    if not data.get("is_follow_up"):
                        st.session_state.question_number = data.get("question_number", st.session_state.question_number + 1)
                    st.rerun()

    # Q&A History accordion
    if st.session_state.qa_history:
        st.markdown("---")
        with st.expander(f"📋 Interview History ({len(st.session_state.qa_history)} questions)", expanded=False):
            for i, item in enumerate(st.session_state.qa_history, 1):
                label = "↩️ Follow-up" if item.get("is_follow_up") else f"Q{i}"
                with st.container():
                    st.markdown(f"**{label}:** {item['question']}")
                    st.markdown(f"**Your Answer:** {item['answer'][:200]}{'...' if len(item['answer']) > 200 else ''}")
                    score = item.get("score", 0)
                    st.markdown(render_score_badge(score), unsafe_allow_html=True)
                    st.markdown("---")


def page_report():
    """Final report page."""
    st.markdown("""
    <div class="main-header">
        <h1>📊 Interview Report</h1>
        <p>Your AI-powered hiring assessment</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.final_report:
        st.error("No report available. Please complete the interview first.")
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
        return

    report = st.session_state.final_report

    # ── Decision Banner ──
    decision = report.get("decision", "N/A")
    score = report.get("overall_score", 0)
    grade = report.get("grade", "N/A")

    decision_colors = {
        "Hire": ("🎉", "hire-badge", "Congratulations!"),
        "No Hire": ("❌", "nohire-badge", "Not selected at this time"),
        "Borderline": ("⚠️", "borderline-badge", "Under Consideration"),
    }
    emoji, badge_class, subtitle = decision_colors.get(decision, ("❓", "borderline-badge", ""))

    col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
    with col_d2:
        st.markdown(f"""
        <div style="text-align:center; padding: 2rem;">
            <div style="font-size: 3rem;">{emoji}</div>
            <div class="{badge_class}">{decision}</div>
            <div style="margin-top:0.5rem; color:#666;">{subtitle}</div>
            <div style="font-size: 2rem; font-weight: bold; margin-top: 1rem;">{score}/10</div>
            <div style="color: #666;">Grade: {grade}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Summary ──
    if report.get("interview_summary"):
        st.markdown("### 📋 Interview Summary")
        st.info(report["interview_summary"])

    if report.get("decision_reasoning"):
        st.markdown("### 💡 Decision Reasoning")
        st.markdown(report["decision_reasoning"])

    # ── Strengths & Weaknesses ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Strengths")
        strengths = report.get("strengths", [])
        if strengths:
            for s in strengths:
                st.success(f"• {s}")
        else:
            st.markdown("_No specific strengths noted._")

    with col2:
        st.markdown("### ❌ Areas for Improvement")
        weaknesses = report.get("weaknesses", [])
        if weaknesses:
            for w in weaknesses:
                st.error(f"• {w}")
        else:
            st.markdown("_No specific weaknesses noted._")

    # ── Technical Areas ──
    if report.get("technical_areas_strong") or report.get("technical_areas_weak"):
        st.markdown("### 🔧 Technical Assessment")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("**Strong Areas:**")
            for area in report.get("technical_areas_strong", []):
                st.markdown(f"✅ {area}")
        with col_t2:
            st.markdown("**Weak Areas:**")
            for area in report.get("technical_areas_weak", []):
                st.markdown(f"📚 {area}")

    # ── Recommendations ──
    st.markdown("### 📈 Recommendations")
    recommendations = report.get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")

    # ── Score Breakdown ──
    st.markdown("---")
    st.markdown("### 📊 Score Breakdown")

    breakdown = report.get("score_breakdown", {})
    individual_scores = report.get("individual_scores", [])

    if individual_scores:
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            st.metric("Overall", f"{score}/10")
        with col_b2:
            st.metric("Highest", f"{breakdown.get('highest', max(individual_scores))}/10")
        with col_b3:
            st.metric("Lowest", f"{breakdown.get('lowest', min(individual_scores))}/10")
        with col_b4:
            total_q = breakdown.get("total_questions", len(individual_scores))
            above_7 = breakdown.get("questions_above_7", sum(1 for s in individual_scores if s >= 7))
            st.metric("Questions ≥7/10", f"{above_7}/{total_q}")

        # Score chart
        if len(individual_scores) > 1:
            import pandas as pd
            chart_data = pd.DataFrame({
                "Question": [f"Q{i+1}" for i in range(len(individual_scores))],
                "Score": individual_scores,
            })
            st.bar_chart(chart_data.set_index("Question"))

    # ── Q&A History ──
    if st.session_state.qa_history:
        st.markdown("---")
        with st.expander("📋 Full Interview Transcript", expanded=False):
            for i, item in enumerate(st.session_state.qa_history, 1):
                label = "↩️ Follow-up" if item.get("is_follow_up") else f"Question {i}"
                st.markdown(f"**{label}:** {item['question']}")
                st.markdown(f"**Your Answer:** {item['answer']}")
                score_val = item.get("score", 0)
                st.markdown(render_score_badge(score_val), unsafe_allow_html=True)
                st.markdown("---")

    # ── Actions ──
    st.markdown("---")
    col_act1, col_act2, col_act3 = st.columns(3)

    with col_act1:
        if st.button("🔁 Start New Interview", use_container_width=True, type="primary"):
            # Reset state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()

    with col_act2:
        report_json = json.dumps(report, indent=2)
        st.download_button(
            "📥 Download Report (JSON)",
            report_json,
            file_name=f"interview_report_{st.session_state.session_id or 'report'}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_act3:
        if st.button("📋 Copy Session ID", use_container_width=True):
            st.code(st.session_state.session_id or "N/A")


# ─────────────────────────────────────────────
# Main Router
# ─────────────────────────────────────────────

def main():
    render_sidebar()

    page = st.session_state.get("page", "home")

    if page == "home":
        page_home()
    elif page == "interview":
        page_interview()
    elif page == "report":
        page_report()
    else:
        page_home()


if __name__ == "__main__":
    main()