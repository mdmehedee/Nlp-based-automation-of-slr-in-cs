# Automated SLR (Computer Science)

AI-assisted prototype for **Systematic Literature Review (SLR)** in Computer Science, built with **Streamlit**.

The app supports a full pipeline from planning to conducting:
- topic and PICOC definition
- research questions
- digital library selection
- inclusion/exclusion criteria
- QA checklist
- data extraction form
- arXiv retrieval, screening, QA scoring, extraction, and taxonomy generation

---

## Features

### Phase A — Planning
1. **Research Topic**
2. **PICOC Framework Definition** (AI suggestions + manual editing)
3. **Research Questions** (AI suggestions + manual editing)
4. **Select Digital Libraries** (arXiv default)
5. **Define Inclusion/Exclusion Criteria** (AI suggestions + manual editing)
6. **Define Quality Assessment Checklist** (AI suggestions + editable weights)
7. **Define Data Extraction Form** (structured editable form, no AI required)

### Phase B — Conducting
1. **Build Search Strings** (base + arXiv format)
2. **Gather Studies (arXiv)** and save raw results
3. **Refinement 1: Dedupe + Screen** (semantic relevance threshold)
4. **Refinement 2: Assign Quality Scores** (Yes=1, Partial=0.5, No=0)
5. **Data Extraction** (AI-assisted per passed paper)
6. **Build Hierarchical Taxonomy** (BERTopic if installed, otherwise TF-IDF fallback)

---

## Project Structure

```text
NLP Based Automation of SLR in CS/
├── src/
│   ├── automated_slr.py      # Main Streamlit app
│   └── agent_setup.py        # OpenAI helpers for PICOC/RQ/criteria/QA/extraction
├── reports/                  # Generated artifacts (protocol, arxiv, screened, taxonomy, etc.)
├── docker/
│   └── Dockerfile
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.11 recommended
- OpenAI-compatible API endpoint and key
- Internet access (OpenAI API + arXiv API)

---

## Environment Variables

Create `.env` in project root:

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://kiste.informatik.tu-chemnitz.de/v1
OPENAI_MODEL=gpt-oss-120b
```

Used by `src/agent_setup.py`.

---

## Run Locally

```bash
cd "/Users/mdmehedeezamankhan/Desktop/NLP Based Automation of SLR in CS"
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run src/automated_slr.py
```

Open: `http://localhost:8501`

---

## Run with Docker

### 1) Standard (faster build, TF-IDF fallback taxonomy)

```bash
cd "/Users/mdmehedeezamankhan/Desktop/NLP Based Automation of SLR in CS"
docker build --no-cache -t automated-slr -f docker/Dockerfile .
docker stop automated-slr 2>/dev/null || true
docker rm automated-slr 2>/dev/null || true
docker run -d --name automated-slr -p 8501:8501 --env-file .env -v "$PWD/reports:/app/reports" automated-slr
docker logs -f automated-slr
```

### 2) With BERTopic + sentence-transformers

```bash
docker build --no-cache -t automated-slr -f docker/Dockerfile --build-arg WITH_BERTOPIC=1 .
docker stop automated-slr 2>/dev/null || true
docker rm automated-slr 2>/dev/null || true
docker run -d --name automated-slr -p 8501:8501 --env-file .env -v "$PWD/reports:/app/reports" automated-slr
docker logs -f automated-slr
```

If BERTopic is not installed, Step 6 automatically uses TF-IDF clustering fallback.

---

## Main Output Files

Generated in `reports/`:

- `protocol.json` — planning protocol (topic, PICOC, RQs, libraries, criteria, QA, extraction form, search strings)
- `arxiv.json` — raw fetched arXiv records
- `arxiv_screened.json` — Step 3 screening decisions
- `arxiv_quality.json` / `arxiv_quality.csv` — Step 4 QA results
- `extracted.json` — Step 5 extracted structured data
- `taxonomy.json` — taxonomy structure
- `taxonomy_hierarchy.html` — taxonomy hierarchy view

---

## Notes

- The app is **Computer-Science scoped** for arXiv querying (`cat:cs.*` constraints in query builder).
- Data extraction uses OpenAI model output plus paper metadata/abstract (and PDF text when available).
- For reproducibility, mount `reports/` as a Docker volume (already shown in commands).

---

## Troubleshooting

- **`docker: command not found`**  
  Docker Desktop is not installed or not running.

- **`exec format error`** on container start  
  Rebuild the image on your machine architecture with `--no-cache`.

- **`localhost:8501` not reachable**  
  Check logs: `docker logs -f automated-slr` and confirm container is running: `docker ps`.

- **Few or zero papers fetched from arXiv**  
  Broaden PICOC terms and regenerate search strings before fetching.
# Nlp-based-automation-of-slr-in-cs
# Nlp-based-automation-of-slr-in-cs
