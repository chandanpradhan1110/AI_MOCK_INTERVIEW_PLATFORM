"""
Sample test scenarios for the AI Interview Platform.
Run this script to verify the full pipeline works end-to-end.

Usage:
    python tests/test_scenario.py
"""
import os
import sys
import json
import time
import requests

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE = "http://localhost:8000/api/v1"

# ─────────────────────────────────────────────
# SAMPLE RESUME TEXT (simulates PDF extraction)
# ─────────────────────────────────────────────

SAMPLE_RESUME_TEXT = """
Priya Sharma
priya.sharma@email.com | +1-650-555-0192 | linkedin.com/in/priyasharma | github.com/priyads

PROFESSIONAL SUMMARY
Data Scientist with 4 years of experience building ML models for e-commerce and fintech.
Specialized in churn prediction, recommendation systems, and NLP pipelines.

SKILLS
Programming: Python, SQL, R, Scala
ML Frameworks: TensorFlow, PyTorch, Scikit-learn, XGBoost, LightGBM
NLP: Hugging Face Transformers, BERT, spaCy, NLTK
Data Tools: Pandas, NumPy, PySpark, Airflow
Cloud: AWS (SageMaker, S3, Lambda), GCP (BigQuery, Vertex AI)
Databases: PostgreSQL, MongoDB, Redis
Visualization: Tableau, Power BI, Matplotlib, Seaborn

EXPERIENCE

Senior Data Scientist | ShopTech Inc. | 2022 - Present
- Built customer churn prediction model using XGBoost achieving 89% AUC, saving $2.1M annually
- Developed real-time product recommendation engine using collaborative filtering + BERT embeddings
- Led A/B testing framework for 15+ product experiments, improving conversion rate by 12%
- Implemented MLOps pipeline using SageMaker + Airflow for automated model retraining

Data Scientist | FinanceFlow Ltd. | 2020 - 2022
- Built credit risk scoring model using Gradient Boosting, reducing default rate by 18%
- Developed NLP pipeline to classify 50K+ customer support tickets daily using BERT
- Created customer segmentation model using K-Means + PCA, identifying 8 distinct segments
- Automated feature engineering pipeline saving 40 hours/week of manual work

PROJECTS

Churn Prediction System (2022)
- End-to-end ML system predicting customer churn 30 days in advance
- Used: XGBoost, SHAP for explainability, Feature Store, REST API deployment
- Result: 89% AUC, 15% reduction in monthly churn rate

Transformer-based Sentiment Analysis (2021)
- Fine-tuned BERT on domain-specific financial news corpus (50K samples)
- Achieved 94% accuracy on 5-class sentiment classification
- Deployed as microservice handling 10K requests/day

Product Recommendation Engine (2023)
- Hybrid recommendation system combining collaborative filtering + content-based
- Used BERT embeddings for item similarity, real-time serving via Redis
- Increased CTR by 23%, average order value by 8%

EDUCATION
M.S. Data Science | Stanford University | 2020
B.Tech Computer Science | IIT Delhi | 2018

CERTIFICATIONS
- AWS Certified Machine Learning Specialty
- Google Professional Data Engineer
- DeepLearning.AI TensorFlow Developer
"""

# ─────────────────────────────────────────────
# SAMPLE JOB DESCRIPTION
# ─────────────────────────────────────────────

SAMPLE_JD_TEXT = """
Senior Data Scientist - NLP & Machine Learning
TechVision AI | San Francisco, CA | Full-time

ABOUT THE ROLE
We are looking for a Senior Data Scientist to join our AI team building next-generation 
NLP and ML solutions. You will work on large-scale language models, recommendation systems,
and production ML pipelines.

KEY RESPONSIBILITIES
- Design and implement NLP models using Transformers, BERT, GPT architectures
- Build and deploy production ML systems at scale (millions of users)
- Develop attention mechanisms and fine-tuning strategies for domain-specific LLMs
- Create robust MLOps pipelines for continuous model training and monitoring
- Lead A/B testing and experimentation frameworks
- Collaborate with engineering to deploy models with <100ms latency requirements
- Mentor junior data scientists

REQUIRED SKILLS & QUALIFICATIONS
- 4+ years of industry experience in data science or machine learning
- Deep expertise in NLP: Transformers, attention mechanisms, BERT, GPT, T5
- Strong Python skills: PyTorch, TensorFlow, Hugging Face Transformers
- Experience with large-scale data processing: Spark, Dask, or similar
- MLOps experience: model versioning, A/B testing, model monitoring
- Knowledge of vector databases and embedding-based retrieval (FAISS, Pinecone)
- Strong understanding of statistical modeling and experimentation
- Experience with cloud platforms: AWS SageMaker or GCP Vertex AI

PREFERRED QUALIFICATIONS
- Experience with LLM fine-tuning and RLHF
- Knowledge of RAG (Retrieval-Augmented Generation) systems
- Contributions to open-source ML projects
- Publications in NLP/ML conferences (NeurIPS, ICML, ACL, EMNLP)

TECHNICAL REQUIREMENTS
- Proficiency in: Python, PyTorch, Transformers, SQL
- Experience with: FAISS, ChromaDB, LangChain, or similar vector search tools
- Understanding of: attention mechanism, positional encoding, tokenization
- Knowledge of: model quantization, distillation, and inference optimization

ABOUT THE TEAM
10-person AI team working on production systems serving 5M+ daily active users.
We ship fast, learn from data, and care deeply about model performance and reliability.

COMPENSATION
$180,000 - $240,000 base + equity + benefits
"""

# ─────────────────────────────────────────────
# Test Helpers
# ─────────────────────────────────────────────

def print_separator(title: str = ""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def check_api_health():
    """Verify API is running."""
    try:
        resp = requests.get(f"{API_BASE.replace('/api/v1', '')}/health", timeout=5)
        resp.raise_for_status()
        print("✅ API is healthy:", resp.json())
        return True
    except Exception as e:
        print(f"❌ API not reachable: {e}")
        print("  Make sure to run: uvicorn api.main:app --reload --port 8000")
        return False


def test_full_interview_text_only():
    """Test 1: Text-only interview (no resume, no JD)."""
    print_separator("TEST 1: Text-Only Interview")

    # Start
    payload = {
        "role": "Data Scientist",
        "candidate_name": "Test Candidate",
        "max_questions": 3,
    }
    resp = requests.post(f"{API_BASE}/interview/start", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    session_id = data["session_id"]
    question = data["data"]["question"]
    print(f"✅ Session started: {session_id[:8]}...")
    print(f"📌 Q1: {question}")

    # Answer loop
    test_answers = [
        "I have 4 years of experience in data science. I specialize in building classification models using XGBoost and neural networks. I've worked on churn prediction, fraud detection, and NLP systems. My approach always starts with understanding the business problem before jumping to modeling.",
        "The bias-variance tradeoff is fundamental to ML. High bias means the model underfits — it's too simple. High variance means overfitting — it memorizes training data. I handle this through cross-validation, regularization techniques like L1/L2, early stopping for neural networks, and ensemble methods like Random Forest which average out variance.",
        "For model deployment, I use FastAPI to serve models as REST endpoints. I containerize with Docker, orchestrate with Kubernetes, and use SageMaker for managed inference. I also set up monitoring for data drift using tools like Evidently AI, and track metrics in MLflow.",
    ]

    current_question = question
    for i, answer in enumerate(test_answers, 1):
        print(f"\n📝 Submitting answer {i}...")
        payload = {"session_id": session_id, "answer": answer}
        resp = requests.post(f"{API_BASE}/interview/answer", json=payload, timeout=45)
        resp.raise_for_status()
        result = resp.json()
        data = result["data"]

        score = data.get("score", 0)
        feedback = data.get("evaluation", {}).get("feedback", "")
        print(f"  Score: {score}/10")
        print(f"  Feedback: {feedback[:100]}...")

        if data.get("is_complete"):
            print("✅ Interview complete!")
            break

        next_q = data.get("next_question", "")
        is_fu = data.get("is_follow_up", False)
        prefix = "[FOLLOW-UP]" if is_fu else f"[Q{i+1}]"
        print(f"  {prefix}: {next_q[:80]}...")
        time.sleep(1)

    # Finalize
    print("\n📊 Generating final report...")
    resp = requests.post(f"{API_BASE}/interview/finalize", params={"session_id": session_id}, timeout=60)
    resp.raise_for_status()
    report = resp.json()["data"]["report"]

    print(f"\n{'='*40}")
    print(f"FINAL REPORT")
    print(f"Decision:      {report.get('decision')}")
    print(f"Overall Score: {report.get('overall_score')}/10")
    print(f"Grade:         {report.get('grade')}")
    print(f"Summary:       {report.get('interview_summary', '')[:150]}...")
    print(f"Strengths:     {report.get('strengths', [])}")
    print(f"Weaknesses:    {report.get('weaknesses', [])}")

    return session_id, report


def test_jd_rag_interview():
    """Test 2: Interview with JD (RAG test)."""
    print_separator("TEST 2: JD-Based RAG Interview")

    payload = {
        "role": "Data Scientist",
        "candidate_name": "RAG Test User",
        "job_description": SAMPLE_JD_TEXT,
        "max_questions": 3,
    }
    resp = requests.post(f"{API_BASE}/interview/start", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    session_id = data["session_id"]
    question = data["data"]["question"]
    has_jd = data["data"].get("has_jd", False)

    print(f"✅ Session: {session_id[:8]}...")
    print(f"📋 JD Indexed: {has_jd}")
    print(f"📌 Q1: {question}")
    print("\n  [JD mentions Transformers, BERT, RAG, FAISS]")
    print("  [Expect questions about attention, transformers, vector search]")

    # Submit a couple answers
    answers = [
        "I have extensive experience with Transformers. I've fine-tuned BERT for text classification achieving 94% accuracy. I understand the attention mechanism deeply — it computes query, key, value matrices and takes a weighted sum based on similarity scores. This allows the model to attend to relevant parts of the input.",
        "For RAG systems, I combine a retrieval component using FAISS for vector similarity search with a generation component like GPT. The process: embed documents, store in FAISS, on query embed the question, retrieve top-k chunks, pass to LLM with context. I've built this for a Q&A system over 100K documents.",
    ]

    for i, answer in enumerate(answers, 1):
        payload = {"session_id": session_id, "answer": answer}
        resp = requests.post(f"{API_BASE}/interview/answer", json=payload, timeout=45)
        resp.raise_for_status()
        result = resp.json()["data"]
        print(f"\n  Ans {i} Score: {result.get('score', 0)}/10")
        if result.get("is_complete"):
            break
        print(f"  Next Q: {result.get('next_question', '')[:80]}...")
        time.sleep(1)

    print("✅ JD RAG test complete")
    return session_id


def test_session_listing():
    """Test 3: List sessions."""
    print_separator("TEST 3: Session Listing")
    resp = requests.get(f"{API_BASE}/sessions?limit=5", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    print(f"✅ Found {data['count']} sessions")
    for s in data["sessions"][:3]:
        print(f"  - {s.get('session_id', '')[:8]}... | {s.get('role')} | {s.get('status')}")


# ─────────────────────────────────────────────
# Sample API Request/Response Documentation
# ─────────────────────────────────────────────

SAMPLE_API_DOCS = """
╔══════════════════════════════════════════════════════════════╗
║           SAMPLE API REQUEST / RESPONSE EXAMPLES            ║
╚══════════════════════════════════════════════════════════════╝

─── 1. START INTERVIEW ───────────────────────────────────────

REQUEST:
POST /api/v1/interview/start
Content-Type: application/json

{
    "role": "Data Scientist",
    "candidate_name": "Priya Sharma",
    "job_description": "Senior DS role requiring Transformers, FAISS...",
    "max_questions": 8
}

RESPONSE:
{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Interview started successfully",
    "data": {
        "question": "Welcome! I've reviewed your background. Let's start — can you give me a brief overview of your experience and what specifically draws you to the Data Scientist role?",
        "question_number": 1,
        "total_questions": 8,
        "difficulty": "medium",
        "is_follow_up": false,
        "role": "Data Scientist"
    }
}

─── 2. SUBMIT ANSWER ─────────────────────────────────────────

REQUEST:
POST /api/v1/interview/answer
Content-Type: application/json

{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "answer": "I have 4 years of experience in data science, specializing in churn prediction and NLP. At ShopTech, I built an XGBoost-based churn model achieving 89% AUC that saved $2.1M annually. I'm drawn to this role because of the focus on Transformers and large-scale ML."
}

RESPONSE:
{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Answer submitted",
    "data": {
        "evaluation": {
            "score": 8.5,
            "feedback": "Excellent response with concrete metrics and business impact. The candidate demonstrated both technical depth and business awareness.",
            "strengths": "Specific quantifiable results ($2.1M, 89% AUC), clear articulation of motivation aligned with role requirements.",
            "improvement": "Could elaborate more on the NLP experience given the role's heavy NLP focus.",
            "technical_accuracy": "Accurately described XGBoost and provided realistic performance metrics."
        },
        "score": 8.5,
        "feedback": "Excellent response...",
        "is_complete": false,
        "questions_answered": 1,
        "total_questions": 8,
        "next_question": "You worked on churn prediction using XGBoost — can you walk me through your feature engineering process and how you handled class imbalance?",
        "question_number": 2,
        "is_follow_up": false,
        "difficulty": "medium"
    }
}

─── 3. FINAL REPORT ──────────────────────────────────────────

REQUEST:
POST /api/v1/interview/finalize?session_id=550e8400-e29b-41d4-a716-446655440000

RESPONSE:
{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Interview finalized",
    "data": {
        "report": {
            "overall_score": 8.1,
            "grade": "A",
            "decision": "Hire",
            "decision_reasoning": "Priya demonstrated strong technical depth across ML, NLP, and MLOps. Her practical experience with Transformers, production deployment, and measurable business impact make her an excellent fit.",
            "strengths": [
                "Deep expertise in NLP and Transformer architectures",
                "Strong track record of delivering business value (churn savings, CTR improvement)",
                "Excellent MLOps knowledge including SageMaker and monitoring"
            ],
            "weaknesses": [
                "Limited discussion of distributed training for very large models",
                "Could strengthen knowledge of RLHF techniques"
            ],
            "recommendations": [
                "Explore LLM fine-tuning with RLHF and PPO",
                "Practice system design for ML systems at 100M+ scale",
                "Deepen knowledge of model quantization and inference optimization"
            ],
            "technical_areas_strong": ["NLP", "MLOps", "Feature Engineering", "A/B Testing"],
            "technical_areas_weak": ["Distributed Training", "RLHF"],
            "interview_summary": "Strong candidate with 4 years of relevant experience...",
            "score_breakdown": {
                "scores": [8.5, 7.0, 9.0, 8.5, 7.5, 8.0, 9.0, 7.5],
                "average": 8.125,
                "highest": 9.0,
                "lowest": 7.0,
                "grade": "A",
                "total_questions": 8,
                "questions_above_7": 8,
                "questions_below_5": 0
            },
            "total_questions": 8,
            "individual_scores": [8.5, 7.0, 9.0, 8.5, 7.5, 8.0, 9.0, 7.5]
        }
    }
}
"""


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🤖 AI MOCK INTERVIEW PLATFORM — TEST SUITE")
    print("=" * 60)

    print(SAMPLE_API_DOCS)

    print_separator("RUNNING LIVE TESTS")

    if not check_api_health():
        print("\n⚠️  Start the API first with:")
        print("    cd ai-interview-app && uvicorn api.main:app --reload --port 8000")
        sys.exit(1)

    try:
        # Run tests
        test_full_interview_text_only()
        time.sleep(2)
        test_jd_rag_interview()
        time.sleep(1)
        test_session_listing()

        print_separator("ALL TESTS PASSED ✅")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)