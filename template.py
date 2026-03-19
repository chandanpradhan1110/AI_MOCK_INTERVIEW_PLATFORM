import os
from pathlib import Path

project_name = "ai-interview-app"

list_of_files = [

    # Agents
    f"{project_name}/agents/__init__.py",
    f"{project_name}/agents/question_agent.py",
    f"{project_name}/agents/evaluator_agent.py",
    f"{project_name}/agents/followup_agent.py",
    f"{project_name}/agents/hr_agent.py",

    # RAG
    f"{project_name}/rag/__init__.py",
    f"{project_name}/rag/retriever.py",
    f"{project_name}/rag/embeddings.py",
    f"{project_name}/rag/chunking.py",

    # Resume
    f"{project_name}/resume/__init__.py",
    f"{project_name}/resume/parser.py",

    # Workflows
    f"{project_name}/workflows/__init__.py",
    f"{project_name}/workflows/interview_graph.py",

    # API
    f"{project_name}/api/__init__.py",
    f"{project_name}/api/main.py",

    # Frontend
    f"{project_name}/frontend/__init__.py",
    f"{project_name}/frontend/streamlit_app.py",

    # Utils
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/prompts.py",
    f"{project_name}/utils/scoring.py",

    # Database
    f"{project_name}/database/__init__.py",
    f"{project_name}/database/models.py",

    # Tests
    f"{project_name}/tests/__init__.py",
    f"{project_name}/tests/test_scenario.py",

    # Root files
    "config.py",
    "requirements.txt",
    ".env",
    "README.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"File already exists at: {filepath}")