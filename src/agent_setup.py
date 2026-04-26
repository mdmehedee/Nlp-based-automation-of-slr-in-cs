import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Prefer environment for endpoint and model; fall back to campus endpoint
# Note: the correct base does NOT include "/openai"; it should end with "/v1"
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://kiste.informatik.tu-chemnitz.de/v1")
MODEL_ID = os.environ.get("OPENAI_MODEL", "gpt-oss-120b")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=BASE_URL)

def get_picoc_from_ai(topic: str):
    """Ask the agent to propose PICOC lists for a given topic."""
    prompt = f"""
    You are assisting in Planning Step 1 of a Systematic Literature Review for the Computer Science domain only.
    Topic: "{topic}".

    Produce PICOC keywords and synonyms modeled after the classic PICOC table used in SLR planning (Population, Intervention, Comparison, Outcome, Context).

    Output requirements (follow strictly):
    - Return ONLY a JSON object with exactly these 5 keys: Population, Intervention, Comparison, Outcome, Context.
    - Each value must be an array of 3 to 5 short noun phrases; 1–3 words per item; no sentences, no punctuation, no emojis.
    - Keep items specific to Computer Science; exclude biomedical/clinical terms (e.g., patient, clinical trial, therapy).
    - Ensure uniqueness (no duplicates, case-insensitive) and consistent casing (Title Case for multi-word phrases, lowercase for metrics like f1-score).
    - Do not wrap in markdown/code fences; return valid JSON only.

    Guidance and scope:
    - Population: Can be a specific role, an application area, or Computer Science domains relevant to the topic (e.g., " which group of people, programs or businesses are of interest for the review?").
    - Intervention: methods, tools, models, or technologies that address the specific issue or topic (e.g., "which technology, tool or procedure is under study?").
    - Comparison: Alternative approaches or baselines, tool, or technology in which the Intervention is being compared (if appropriate) (e.g., "how is the control treatment defined? In particular the ‘placebo’ intervention is critical, as “not using the intervention” is mostly not a valid action in software engineering").
    - Outcome: Measurable results and qualities (e.g., "The outcomes of the experiment should not only be statistically significant, but also be significant from a practical point of view. For example, it is probably not interesting that an outcome is 10% better in some respect if it is twice as time consuming.").
    - Context: The context in which the comparison takes place and sources where studies occur (e.g., "which is an extended view of the population, including whether it is conducted in academia or industry, in which industry segment, and also the incentives for the subjects.").

    Example structure (illustrative only; adapt to the given topic):
    {{
      "Population": ["Researchers","Software Engineers", "Smart Manufacturing", "Data Scientists", "Autonomous Driving"],
      "Intervention": ["Semantic Web", "Transformer Models", "Graph Neural Networks", "Information Retrieval"],
      "Comparison": ["Rule-based Methods", "Classical ML", "Manual Screening", "Keyword Search", "Heuristic Baseline"],
      "Outcome": ["Accuracy", "Precision", "Recall", "F1-score", "Latency", "Throughput", "Scalability", "Robustness"],
      "Context": ["Digital Libraries", "Academic Conferences", "Code Repositories", "Benchmark Datasets", "Industry Case Studies"]
    }}
    """
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You write only valid JSON unless otherwise instructed."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or "{}"


def get_rqs_from_ai(topic: str, picoc: dict):
    """Generate 2–3 concise CS-focused research questions from PICOC.

    Returns a JSON array of strings. No numbering or bullets.
    """
    # Light guard to avoid huge prompts
    safe_picoc = {k: v[:10] if isinstance(v, list) else v for k, v in (picoc or {}).items()}
    prompt = f"""
    You are assisting with Planning Step 3 of a Systematic Literature Review in Computer Science.
    Topic: "{topic}".
    PICOC terms (cleaned): {safe_picoc}

    Task: Propose 2 to 3 precise research questions that will guide study identification and data extraction.
    Requirements:
    - Write clear, answerable questions grounded in the PICOC elements.
    - Prefer formulations like: "What are...", "Which ... is/are ...?", "In which scenarios ...?".
    - Domain: Computer Science only.
    - Keep each question under 180 characters.
    - Output ONLY a JSON array of 2–3 strings; no numbering, no markdown, no extra text.
    """
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You write only valid JSON unless otherwise instructed."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or "[]"


def get_criteria_from_ai(topic: str, picoc: dict | None = None, rqs: list | None = None, libraries: list | None = None):
    """Suggest inclusion and exclusion criteria for the SLR planning phase.

    Returns a JSON object with exactly these keys and structure (keep each list concise):
    {
      "Period": {"include": [..], "exclude": [..]},
      "Language": {"include": [..], "exclude": [..]},
      "Type of Literature": {"include": [..], "exclude": [..]},
      "Type of Source": {"include": [..], "exclude": [..]},
      "Impact Source": {"include": [..], "exclude": [..]},
      "Accessibility": {"include": [..], "exclude": [..]},
      "Relevance to RQs": {"include": [..], "exclude": [..]}
    }
    """
    safe_picoc = {k: v[:8] if isinstance(v, list) else v for k, v in (picoc or {}).items()}
    libs = ", ".join(libraries or [])
    safe_rqs = (rqs or [])[:3]
    prompt = f"""
    You are assisting with SLR Planning Step 5: define inclusion and exclusion criteria.
    Topic: "{topic}"
    PICOC (truncated): {safe_picoc}
    Research Questions (truncated): {safe_rqs}
    Selected libraries: {libs}

    Produce concise, practical criteria for Computer Science, grounded in the Topic, PICOC, and selected libraries
    (e.g., if arXiv is selected, prefer items about preprints and CS categories). Prefer items that can be decided
    from abstract/metadata. Return ONLY valid JSON with exactly these keys, each having "include" and "exclude"
    lists of short phrases (aim for 2 per category; do not exceed 4 per category):
    Period, Language, Type of Literature, Type of Source, Impact Source, Accessibility, Relevance to RQs.
    Keep items brief (3–8 items per list). No prose or markdown.
    """
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You write only valid JSON unless otherwise instructed."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or "{}"


def get_qa_checklist_from_ai(topic: str, picoc: dict | None = None, rqs: list | None = None):
    """Suggest 4–6 concise QA questions for a CS SLR checklist, with optional weights.

    Expected output (strict): JSON array where each item is either a string question
    or an object {"text": "...", "weight": 1 | 0.5 | 0}. No markdown, no extra text.
    """
    safe_picoc = {k: v[:8] if isinstance(v, list) else v for k, v in (picoc or {}).items()}
    safe_rqs = (rqs or [])[:3]
    prompt = f"""
    You are assisting with Planning Step 6: define a Quality Assessment (QA) checklist for a Systematic Literature Review in Computer Science.
    Topic: "{topic}"
    PICOC (truncated): {safe_picoc}
    Research Questions (truncated): {safe_rqs}

    Task: Propose 4 to 6 short, objective yes/no questions that assess Reporting, Rigor, Credibility, and Relevance of primary studies in Computer Science. Avoid domain-irrelevant or biomedical phrasing. Keep each item under 120 characters.

    Output strictly: return ONLY a JSON array of objects, each with keys "text" and "weight" where weight ∈ {{1, 0.5, 0}}.
    Example format (illustrative only):
    [
      {{"text": "Are the research aims clearly stated?", "weight": 1}},
      {{"text": "Are data sources described?", "weight": 0.5}}
    ]

    Do not include numbering, markdown, comments, or any other text besides the JSON array.
    """
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You write only valid JSON unless otherwise instructed."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or "[]"


def get_extraction_form_from_ai(topic: str, picoc: dict | None = None, rqs: list | None = None):
    """Propose 6–10 data extraction fields tailored to the SLR in CS.

    Expected output (strict): a JSON array of field objects with keys:
      - name: short label
      - type: one of [text, long_text, number, boolean, select, multiselect]
      - options: optional array of strings (for select/multiselect)
      - help: optional short hint
    """
    safe_picoc = {k: v[:8] if isinstance(v, list) else v for k, v in (picoc or {}).items()}
    safe_rqs = (rqs or [])[:3]
    prompt = f"""
    You are assisting with Planning Step 7: define a Data Extraction form for a Systematic Literature Review in Computer Science.
    Topic: "{topic}"
    PICOC (truncated): {safe_picoc}
    Research Questions (truncated): {safe_rqs}

    Task: Propose 6 to 10 fields to extract from each included study that help answer the RQs. Cover typical CS SLR facets such as:
    - Research type (Theoretical / Empirical),
    - Process phases/stages,
    - Technology / framework / platform,
    - Application field / domain,
    - Findings and outcomes (metrics),
    - Gaps/challenges,
    - Evaluation method (dataset, benchmark, case study),
    - Year, Venue.

    Output strictly: ONLY a JSON array of objects with keys:
      "name" (short label),
      "type" in ["text","long_text","number","boolean","select","multiselect"],
      optional "options" (array of strings for select/multiselect),
      optional "help" (short hint).
    Example format (illustrative only):
    [
      {{"name":"Research type","type":"select","options":["Theoretical","Empirical"],"help":"overall study category"}},
      {{"name":"Technology","type":"text"}}
    ]
    Do not include markdown or comments—return the JSON array only.
    """
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You write only valid JSON unless otherwise instructed."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or "[]"


def extract_data_for_paper(
    topic: str,
    rqs: list | None,
    form_fields: list[dict],
    title: str,
    abstract: str,
    extra_text: str | None = None,
):
    """Ask the model to fill the data extraction form for one paper.

    Returns ONLY a JSON object mapping field name -> short string value.
    """
    # Keep the context reasonably short
    def _trim(txt: str, limit: int = 6000) -> str:
        if not txt:
            return ""
        txt = str(txt)
        return txt[:limit]

    fields = []
    for f in form_fields or []:
        name = str(f.get("name", "")).strip()
        if name:
            fields.append(name)

    prompt = f"""
    You help with a Computer Science SLR. Given the paper text, fill a structured data extraction form.

    Topic: {topic}
    Research Questions: {rqs or []}

    Paper metadata:
    - Title: {title}
    - Abstract: {abstract}
    - Extra text (may be truncated): { _trim(extra_text or '') }

    Output strictly: return ONLY a JSON object whose keys are exactly the following field names and whose values are concise strings (max ~200 chars each). If unknown, use an empty string "".
    Field names: {fields}
    """
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You write only valid JSON unless otherwise instructed."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or "{}"
