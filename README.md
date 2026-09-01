# Multi-Agent Argumentative Debate & Decision Engine

An **Adversarial Consensus Reasoning Engine** built with Python, LangChain, ChromaDB, and Streamlit. The system orchestrates multi-agent debate to rigorously test premises, identify logical flaws, verify document-grounded claims, and reach structured, objective consensus.

---

## 📌 Project Overview

Traditional single-prompt LLM reasoning often suffers from confirmation bias, logical fallacies, and hallucinated claims. The **Multi-Agent Argumentative Debate & Decision Engine** addresses this by orchestrating an adversarial debate between specialized AI agents:

1. **Proponent (Agent Alpha)**: Formulates and defends a thesis using document-grounded evidence.
2. **Opponent (Agent Beta)**: Aggressively challenges Alpha's thesis, searching for logical gaps, unsupported claims, and contradictions.
3. **Moderator (Orchestrator)**: Evaluates arguments impartially, detects logical fallacies, verifies citations against local RAG evidence, and synthesizes structured final consensus using Pydantic.

---

## 🏗️ Architecture Overview

The system is split into two primary decoupled layers:

### 1. Document Grounding (RAG Pipeline)
```
Uploaded Document ──► Document Loader ──► Text Chunker ──► Embeddings ──► ChromaDB Vector Store ──► Retriever ──► Grounding Verification
```

### 2. Multi-Agent Debate Loop
```
User Premise ──► Debate State ──► Proponent (Alpha) ──► Opponent (Beta) ──► Moderator (Judge) ──► Consensus / Next Round
```

For complete technical specifications, see [docs/architecture.md](docs/architecture.md).

---

## 🛠️ Technology Stack

- **Language**: Python 3.10+
- **Orchestration**: LangChain
- **Vector Database**: ChromaDB
- **Data Schemas & Validation**: Pydantic v2
- **UI Framework**: Streamlit
- **Environment Management**: `python-dotenv` & `pydantic-settings`
- **Testing**: Pytest

---

## 📂 Repository Structure

```
Multi-Agent-Debat_Engine/
│
├── README.md                  # Project documentation and developer guide
├── .gitignore                 # Version control exclusions
├── requirements.txt           # Dependency manifests
├── .env.example               # Template environment configuration
│
├── src/                       # Source package
│   ├── __init__.py
│   ├── agents/                # AI Agent definitions (Proponent, Opponent, Moderator)
│   ├── rag/                   # Document loader, chunker, embeddings, vector store, retriever, grounding
│   ├── debate/                # State definitions, debate manager, and multi-agent orchestrator
│   ├── config/                # Settings and environment configuration
│   └── utils/                 # Logging and helper utilities
│
├── app/                       # User Interface
│   └── streamlit_app.py       # Streamlit web interface
│
├── tests/                     # Unit test suite
│   ├── test_rag.py            # RAG pipeline tests
│   ├── test_grounding.py      # Metadata preservation and grounding tests
│   └── test_agents.py         # Agent schema and interaction tests
│
├── data/                      # Local vector DB and uploaded files directory (.gitignored)
│   └── .gitkeep
│
└── docs/                      # Technical documentation
    └── architecture.md        # Detailed architecture and flow diagrams
```

---

## 🚀 Development Phases

- **Phase 1**: Core multi-agent orchestration + basic Streamlit UI skeleton *(Current Phase)*
- **Phase 2**: RAG integration with local ChromaDB, PDF parsing, and metadata preservation
- **Phase 3**: Fine-grained moderation, Pydantic structured output, and UI polish
- **Phase 4**: Evaluation metrics, empirical testing, and exhibition demo

---

## ⚙️ Setup & Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_ORGANIZATION/Multi-Agent-Debat_Engine.git
cd Multi-Agent-Debat_Engine
```

### 2. Set Up a Virtual Environment
```bash
# MacOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and configure your local settings:
```bash
cp .env.example .env
```
*(Note: Never commit `.env` containing actual API keys to GitHub!)*

---

## 🌿 Git Branching & Collaboration Guidelines

To maintain repository quality and avoid git conflicts across team members:

1. **Never commit directly to `main`.**
2. Always create a feature branch named after your feature area:
   - Agent logic: `feature/agent-proponent`, `feature/agent-opponent`, `feature/moderator`
   - RAG logic: `feature/rag-loader`, `feature/chromadb-integration`
   - UI: `feature/streamlit-ui`
   - Docs & Tests: `docs/architecture-update`, `test/agent-tests`
3. **Workflow Command Example**:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/agent-proponent
   
   # Make changes and commit
   git add src/agents/proponent.py
   git commit -m "feat(agents): add ProponentAgent stub and prompt interface"
   git push origin feature/agent-proponent
   ```
4. Open a Pull Request (PR) on GitHub for team review before merging into `main`.

---

## 🧪 Running Tests and Application

### Run Tests
```bash
pytest tests/
```

### Run Streamlit App
```bash
streamlit run app/streamlit_app.py
```
