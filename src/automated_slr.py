import json
from pathlib import Path
import shutil
from datetime import datetime
import streamlit as st
from agent_setup import (
    get_picoc_from_ai,
    get_rqs_from_ai,
    get_criteria_from_ai,
    get_qa_checklist_from_ai,
    get_extraction_form_from_ai,
    extract_data_for_paper,
)


st.set_page_config(page_title="Automated SLR", page_icon="📝", layout="centered")

reports_dir = Path("reports")
reports_dir.mkdir(parents=True, exist_ok=True)
protocol_path = reports_dir / "protocol.json"

PICOC_KEYS = ["Population", "Intervention", "Comparison", "Outcome", "Context"]

# ArXiv CS categories to restrict results to Computer Science domain
CS_CATEGORIES = [
    "cs.AI", "cs.CL", "cs.CC", "cs.CE", "cs.CG", "cs.CR", "cs.CV", "cs.CY",
    "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL",
    "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA",
    "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF",
    "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY",
]


# ---------- Helpers ----------
def dedup_preserve(seq):
    seen = set()
    out = []
    for item in seq or []:
        key = str(item).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(str(item).strip())
    return out


def _safe_read_json(path: Path, default):
    """Best-effort JSON reader with small retries; never raises to UI."""
    try:
        import time
    except Exception:
        time = None
    for _ in range(3):
        try:
            if not path.exists():
                return default
            text = path.read_text(encoding="utf-8")
            if not (text or "").strip():
                return default
            return json.loads(text)
        except Exception:
            if time:
                time.sleep(0.05)
            continue
    return default


def _safe_write_json(path: Path, data) -> None:
    """Atomic-ish write: write to temp then replace; swallow IO errors."""
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass


def _quote(term: str) -> str:
    t = (term or "").strip().replace('"', '\\"')
    return f'"{t}"'


def _or_group(terms):
    cleaned = dedup_preserve([t for t in (terms or []) if isinstance(t, str)])
    if not cleaned:
        return ""
    return "(" + " OR ".join(_quote(t) for t in cleaned) + ")"


def build_base_query(picoc: dict) -> str:
    parts = []
    for key in PICOC_KEYS:
        grp = _or_group((picoc or {}).get(key, []))
        if grp:
            parts.append(grp)
    return " AND ".join(parts) if parts else ""


def build_arxiv_query(picoc: dict) -> str:
    def fielded_group(terms):
        cleaned = dedup_preserve([t for t in (terms or []) if isinstance(t, str)])
        if not cleaned:
            return ""
        return "(" + " OR ".join(f'all:{_quote(t)}' for t in cleaned) + ")"

    parts = []
    for key in PICOC_KEYS:
        grp = fielded_group((picoc or {}).get(key, []))
        if grp:
            parts.append(grp)
    # Always constrain to Computer Science categories on arXiv
    cat_clause = "(" + " OR ".join(f"cat:{c}" for c in CS_CATEGORIES) + ")"
    if parts:
        parts.append(cat_clause)
    else:
        parts = [cat_clause]
    return " AND ".join(parts)


def convert_base_to_arxiv(base_str: str) -> str:
    import re
    if not base_str:
        return ""

    def repl(m):
        token = m.group(0)
        return token if token.startswith("all:") else f"all:{token}"

    return re.sub(r'(?<![A-Za-z]:)"[^"\\]*(?:\\.[^"\\]*)*"', lambda m: repl(m), base_str)


# ---------- Planning Phase ----------
BOOT_CLEANED = False


def _cold_boot_reset():
    global BOOT_CLEANED
    if BOOT_CLEANED:
        return
    # If no session topic, ensure protocol.json starts empty; backup any previous content
    if "topic" not in st.session_state:
        try:
            if protocol_path.exists():
                try:
                    content = (protocol_path.read_text(encoding="utf-8").strip() or "")
                except Exception:
                    content = ""
                if content and content not in ("{}", "[]"):
                    history = reports_dir / "history"
                    history.mkdir(exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    shutil.copy2(protocol_path, history / f"protocol-{ts}.json")
                    _safe_write_json(protocol_path, {})
        except Exception:
            # Never block the UI on backup errors
            pass
    BOOT_CLEANED = True


def render_planning():
    st.markdown("Planning Phase")

    # Cold boot: reset file (backed up) and clear volatile state
    _cold_boot_reset()
    if "topic" not in st.session_state:
        for k in ("picoc", "rqs", "base_query_text", "arxiv_query_text", "last_base_snapshot"):
            st.session_state.pop(k, None)

    # Step 1 — Topic input
    st.header("Research Topic")
    topic = st.text_input("Enter your topic (e.g., Sorting Algorithms)", value=st.session_state.get("topic", ""))
    if st.button("Search"):
        if not topic.strip():
            st.error("Please enter a topic.")
        else:
            st.session_state["topic"] = topic.strip()
            # Only record the topic to file (no auto-loading of previous data)
            prot = _safe_read_json(protocol_path, {})
            prot.update({
                "topic": st.session_state["topic"],
                "picoc": prot.get("picoc", {}),
                "rqs": prot.get("rqs", []),
                "base_query": prot.get("base_query", ""),
                "arxiv_query": prot.get("arxiv_query", ""),
            })
            _safe_write_json(protocol_path, prot)
            st.success("Saved topic to reports/protocol.json")
            st.json({"topic": st.session_state["topic"]})

    # Step 2 — PICOC + Synonyms
    st.header("PICOC Framework Definition")
    if st.button("Generate PICOC Framework"):
        if not st.session_state.get("topic"):
            st.error("Need to complete Topic name")
        else:
            st.info("Asking AI Agent for PICOC...")
            try:
                result = get_picoc_from_ai(st.session_state["topic"])
                picoc_data = json.loads(result)
            except Exception as e:
                st.error(f"AI Agent failed: {e}")
                st.stop()
            # Add a broad population anchor to help recall in arXiv queries
            try:
                pops = list(picoc_data.get("Population", []) or [])
                if "researcher" not in [str(x).strip().lower() for x in pops]:
                    pops.append("researcher")
                picoc_data["Population"] = dedup_preserve(pops)
            except Exception:
                pass
            st.session_state["picoc"] = picoc_data

    # PICOC chip editor
    def _remove_term(section_key: str, idx: int):
        terms = st.session_state["picoc"].get(section_key, [])
        if 0 <= idx < len(terms):
            del terms[idx]
            st.session_state["picoc"][section_key] = terms

    def _add_term():
        target_key = st.session_state.get("add_field_select") or PICOC_KEYS[0]
        new_term = (st.session_state.get("add_field_input") or "").strip()
        if not new_term:
            return
        cur = st.session_state["picoc"].get(target_key, [])
        st.session_state["picoc"][target_key] = dedup_preserve(cur + [new_term])
        st.session_state["add_field_input"] = ""

    if "picoc" in st.session_state:
        st.subheader("Suggested Terms")

        def render_chips(section_key: str):
            terms = st.session_state["picoc"].get(section_key, [])
            terms = terms if isinstance(terms, list) else []
            cols = st.columns(4)
            for i, term in enumerate(terms):
                cols[i % 4].button(
                    f"{term} ✕",
                    key=f"chip-{section_key}-{i}",
                    on_click=_remove_term,
                    args=(section_key, i),
                )

        @st.fragment
        def picoc_editor():
            for k in PICOC_KEYS:
                st.markdown(f"**{k} Synonyms:**")
                render_chips(k)
            st.divider()
            st.subheader("Add Terms")
            c1, c2, c3 = st.columns([1, 3, 0.6])
            with c1:
                st.selectbox("Field", PICOC_KEYS, key="add_field_select", label_visibility="collapsed")
            with c2:
                st.text_input("Add synonym...", key="add_field_input", placeholder="Add synonym...", label_visibility="collapsed")
            with c3:
                st.button("➕", key="add_btn", on_click=_add_term)

        picoc_editor()

        if st.button("Save PICOC"):
            prot = _safe_read_json(protocol_path, {})
            prot.setdefault("topic", st.session_state.get("topic", ""))
            prot["picoc"] = {k: st.session_state["picoc"].get(k, []) for k in PICOC_KEYS}
            _safe_write_json(protocol_path, prot)
            st.success("PICOC saved to reports/protocol.json")
            st.json(prot["picoc"])
    else:
        st.info("Click 'Generate PICOC' to see suggestions.")

    # Step 3 — Research questions
    st.header("Research Questions")
    if st.session_state.get("picoc"):
        c1, c2 = st.columns([1, 3])
        with c1:
            suggest = st.button("Generate RQs")
        with c2:
            pass
        if suggest:
            try:
                raw = get_rqs_from_ai(st.session_state.get("topic", ""), st.session_state.get("picoc", {}))
                rqs = json.loads(raw)
                if not isinstance(rqs, list):
                    raise ValueError("AI did not return a list")
                rqs = [str(x).strip() for x in rqs if str(x).strip()][:3]
                while len(rqs) < 2:
                    rqs.append("")
                st.session_state["rqs"] = rqs
            except Exception as e:
                st.error(f"AI Agent failed to suggest RQs: {e}")
    if "rqs" in st.session_state:
        st.subheader("Edit Research Questions")
        new_rqs = []
        base_len = max(2, len(st.session_state["rqs"]))
        for i in range(base_len):
            placeholder = st.session_state["rqs"][i] if i < len(st.session_state["rqs"]) else ""
            new_rqs.append(st.text_input(f"RQ{i+1}", value=placeholder, key=f"rq_{i}"))
        if len(new_rqs) == 2:
            new_rqs.append(st.text_input("RQ3", value="", key="rq_2_extra"))
        if st.button("Save RQs"):
            cleaned = [rq.strip() for rq in new_rqs if rq and rq.strip()]
            st.session_state["rqs"] = cleaned[:3]
            prot = _safe_read_json(protocol_path, {})
            prot.setdefault("topic", st.session_state.get("topic", ""))
            prot["picoc"] = st.session_state.get("picoc", prot.get("picoc", {}))
            prot["rqs"] = st.session_state["rqs"]
            _safe_write_json(protocol_path, prot)
            st.success("RQs saved to reports/protocol.json")
            st.json(prot["rqs"])
    else:
        st.caption("Generate PICOC first, then ask for RQs.")

    # Step 4 — Select digital libraries
    st.header("Select digital libraries")
    lib_catalog = [
        ("arxiv", "arXiv", "https://arxiv.org", "Open preprints; CS categories available"),
        ("ieee", "IEEE Xplore", "https://ieeexplore.ieee.org", "Engineering & technology"),
        ("acm", "ACM Digital Library", "https://dl.acm.org/", "Computing & IT"),
        ("scopus", "Scopus", "https://www.scopus.com/", "Interdisciplinary"),
        ("wos", "Web of Science", "https://www.webofscience.com/", "Interdisciplinary"),
        ("ei", "EI Compendex", "https://www.engineeringvillage.com/", "Engineering"),
    ]
    # default to arXiv only if nothing chosen in this session
    current_libs = st.session_state.get("libraries") or ["arxiv"]
    labels = {k: label for k, label, *_ in lib_catalog}
    options_display = [label for _, label, *_ in lib_catalog]
    value_display = [labels[k] for k in current_libs if k in labels]
    picked_display = st.multiselect(
        "Pick one or more libraries (default: arXiv)",
        options=options_display,
        default=value_display,
    )
    # map labels back to canonical keys
    name_to_key = {label: key for key, label, *_ in lib_catalog}
    picked_keys = [name_to_key[l] for l in picked_display]
    if st.button("Save Libraries"):
        st.session_state["libraries"] = picked_keys
        prot = _safe_read_json(protocol_path, {})
        prot.setdefault("topic", st.session_state.get("topic", ""))
        prot["libraries"] = picked_keys
        _safe_write_json(protocol_path, prot)
        st.success("Libraries saved to reports/protocol.json")
        st.json({"libraries": picked_keys})

    # Step 5 — Define inclusion and exclusion criteria
    st.header("Define inclusion and exclusion criteria")
    st.caption("Set the rules to include or exclude papers. You can refine these later during the Conducting phase.")

    # Only show the button until you generate suggestions
    if st.button("Generate AI suggestions"):
        if not (st.session_state.get("topic") and st.session_state.get("picoc") and st.session_state.get("rqs")):
            st.error("Please complete Topic, PICOC and RQs first.")
        else:
            try:
                raw = get_criteria_from_ai(
                    st.session_state.get("topic", ""),
                    st.session_state.get("picoc", {}),
                    st.session_state.get("rqs", []),
                    st.session_state.get("libraries", ["arxiv"]),
                )
                crit = json.loads(raw) if raw else {}
                inc, exc = [], []
                for v in (crit or {}).values():
                    inc += v.get("include", []) or []
                    exc += v.get("exclude", []) or []
                inc = dedup_preserve(inc)[:10]
                exc = dedup_preserve(exc)[:10]
                st.session_state["crit_list_inc_GLOBAL"] = [{"text": s, "checked": True} for s in inc]
                st.session_state["crit_list_exc_GLOBAL"] = [{"text": s, "checked": True} for s in exc]
                st.session_state["criteria_ui_ready"] = True
            except Exception as e:
                # Handle Streamlit control-flow exceptions quietly
                cls = getattr(e, "__class__", None)
                name = getattr(cls, "__name__", "")
                libs = st.session_state.get("libraries", ["arxiv"]) or []
                lib_phrase = ", ".join(libs)
                inc = [
                    "Within Computer Science or related domain",
                    "Aligns with PICOC scope",
                    "Written in English",
                    f"Published in selected sources ({lib_phrase})",
                    "Accessible (abstract/full text available)",
                ]
                exc = [
                    "Out of scope relative to PICOC",
                    "Non-CS domain without CS relevance",
                    "Duplicate versions of the same paper",
                    "Non-research material (editorial/poster/tutorial/keynote/blog)",
                    "Insufficient information (missing abstract/full text)",
                ]
                st.session_state["crit_list_inc_GLOBAL"] = [{"text": s, "checked": True} for s in inc]
                st.session_state["crit_list_exc_GLOBAL"] = [{"text": s, "checked": True} for s in exc]
                st.session_state["criteria_ui_ready"] = True
                if "RerunData" not in name and "StopException" not in name:
                    st.warning(f"AI suggestions unavailable, loaded defaults. ({e})")

    saved_prot = _safe_read_json(protocol_path, {})
    saved_global = saved_prot.get("criteria_global") or {}
    ui_ready = st.session_state.get("criteria_ui_ready") or bool(saved_global)
    if not ui_ready:
        st.stop()

    if saved_global and not st.session_state.get("crit_list_inc_GLOBAL"):
        st.session_state["crit_list_inc_GLOBAL"] = [{"text": s, "checked": True} for s in saved_global.get("include", [])]
        st.session_state["crit_list_exc_GLOBAL"] = [{"text": s, "checked": True} for s in saved_global.get("exclude", [])]

    # Time window (optional)
    cfy, cty = st.columns(2)
    st.session_state.setdefault("period_from", 2015)
    st.session_state.setdefault("period_to", 2025)
    st.session_state["period_from"] = cfy.number_input("From year", min_value=1900, max_value=2100, value=int(st.session_state["period_from"]), step=1)
    st.session_state["period_to"]   = cty.number_input("To year",   min_value=1900, max_value=2100, value=int(st.session_state["period_to"]),   step=1)

    inc_col, exc_col = st.columns(2)
    with inc_col:
        st.subheader("Inclusion criteria")
        inc_rows = st.session_state.setdefault("crit_list_inc_GLOBAL", [])
        for i, row in enumerate(inc_rows):
            r1, r2 = st.columns([0.1, 0.9])
            row["checked"] = r1.checkbox("", value=row.get("checked", True), key=f"inc_chk_GLOBAL_{i}")
            row["text"]    = r2.text_input("", value=row.get("text", ""), key=f"inc_txt_GLOBAL_{i}")
        add_inc = st.text_input("Custom inclusion", key="custom_inc_GLOBAL")
        if st.button("Add inclusion", key="custom_btn_inc_GLOBAL") and (add_inc or "").strip():
            inc_rows.append({"text": add_inc.strip(), "checked": True})

    with exc_col:
        st.subheader("Exclusion criteria")
        exc_rows = st.session_state.setdefault("crit_list_exc_GLOBAL", [])
        for j, row in enumerate(exc_rows):
            r1, r2 = st.columns([0.1, 0.9])
            row["checked"] = r1.checkbox("", value=row.get("checked", True), key=f"exc_chk_GLOBAL_{j}")
            row["text"]    = r2.text_input("", value=row.get("text", ""), key=f"exc_txt_GLOBAL_{j}")
        add_exc = st.text_input("Custom exclusion", key="custom_exc_GLOBAL")
        if st.button("Add exclusion", key="custom_btn_exc_GLOBAL") and (add_exc or "").strip():
            exc_rows.append({"text": add_exc.strip(), "checked": True})

    if st.button("Save Criteria"):
        include_final = [r.get("text", "").strip() for r in st.session_state.get("crit_list_inc_GLOBAL", []) if r.get("checked") and r.get("text", "").strip()]
        exclude_final = [r.get("text", "").strip() for r in st.session_state.get("crit_list_exc_GLOBAL", []) if r.get("checked") and r.get("text", "").strip()]
        fy = st.session_state.get("period_from")
        ty = st.session_state.get("period_to")
        if fy and ty:
            include_final = [f"From {int(fy)} to {int(ty)}"] + include_final
        prot = _safe_read_json(protocol_path, {})
        prot.setdefault("topic", st.session_state.get("topic", ""))
        prot["criteria_global"] = {"include": include_final, "exclude": exclude_final}
        _safe_write_json(protocol_path, prot)
        st.success("Criteria saved to reports/protocol.json")
        st.json({"criteria_global": prot["criteria_global"]})

    # (Search strings moved to Conducting → Step 1.)

    # Step 6 — Define quality assessment checklist
    st.header("Define quality assessment checklist")
    st.caption("Create questions and weights; scoring is fixed: Yes=1, Partial=0.5, No=0.")

    # Only show generator if topic+PICOC exist
    c1, c2 = st.columns([1, 3])
    with c1:
        gen_qa = st.button("Generate AI checklist")
    with c2:
        pass
    if gen_qa:
        if not (st.session_state.get("topic") and st.session_state.get("picoc")):
            st.error("Please complete Topic and PICOC first.")
        else:
            try:
                raw = get_qa_checklist_from_ai(
                    st.session_state.get("topic", ""),
                    st.session_state.get("picoc", {}),
                    st.session_state.get("rqs", []),
                )
                qs = json.loads(raw)
                if not isinstance(qs, list):
                    raise ValueError("AI did not return a list")
                rows = []
                for item in qs:
                    if isinstance(item, dict):
                        text = str(item.get("text", "")).strip()
                        w = float(item.get("weight", 1.0)) if item.get("weight") is not None else 1.0
                    else:
                        text = str(item).strip()
                        w = 1.0
                    if not text:
                        continue
                    # clamp weight to allowed set {1.0, 0.5, 0.0}
                    w = 1.0 if abs(w - 1.0) < 1e-6 else (0.5 if abs(w - 0.5) < 1e-6 else (0.0 if abs(w - 0.0) < 1e-6 else 1.0))
                    rows.append({"text": text, "weight": w})
                # keep 4–6 items
                rows = rows[:6]
                if len(rows) < 4:
                    raise ValueError("Too few AI questions")
                st.session_state["qa_questions"] = rows
            except Exception as e:
                st.warning(f"AI checklist unavailable, loaded defaults. ({e})")
                defaults = [
                    "Are the research aims clearly stated?",
                    "Is the study design appropriate to the aims?",
                    "Are data sources and collection methods described?",
                    "Are variables/measures clearly defined?",
                    "Are analysis methods described and justified?",
                    "Are threats to validity or limitations discussed?",
                    "Are results reported with sufficient detail?",
                ]
                st.session_state["qa_questions"] = [{"text": q, "weight": 1.0} for q in defaults[:6]]

    # Load saved QA if present and session empty
    saved = _safe_read_json(protocol_path, {})
    saved_qac = saved.get("qa_checklist") or {}
    if saved_qac and not st.session_state.get("qa_questions"):
        rows = saved_qac.get("questions") or []
        st.session_state["qa_questions"] = [{"text": r.get("text", ""), "weight": float(r.get("weight", 1.0))} for r in rows]
        st.session_state["qa_cutoff"] = float(saved_qac.get("cutoff", 0.0))

    # Only render editor if we have questions in session
    qa_rows = st.session_state.get("qa_questions") or []
    if qa_rows:
        st.subheader("Checklist questions")
        # Editable rows with delete buttons
        to_delete = []
        for i, row in enumerate(qa_rows):
            c1, c2, c3 = st.columns([0.08, 0.72, 0.20])
            with c1:
                if st.button("✕", key=f"qa_del_{i}"):
                    to_delete.append(i)
            with c2:
                row["text"] = st.text_input("", value=row.get("text", ""), key=f"qa_txt_{i}", label_visibility="collapsed")
            with c3:
                row["weight"] = st.radio(
                    "Weight",
                    options=[1.0, 0.5, 0.0],
                    index={1.0: 0, 0.5: 1, 0.0: 2}.get(float(row.get("weight", 1.0)), 0),
                    format_func=lambda x: str(x),
                    horizontal=True,
                    key=f"qa_w_{i}",
                    label_visibility="collapsed",
                )
        if to_delete:
            for idx in sorted(to_delete, reverse=True):
                del qa_rows[idx]
            st.session_state["qa_questions"] = qa_rows

        # Add new question row
        # If previous add requested a clear, do it BEFORE rendering the input widget
        if st.session_state.pop("qa_add_clear", False):
            st.session_state["qa_new_q"] = ""
        c_add1, c_add2, c_add3 = st.columns([0.72, 0.20, 0.08])
        with c_add1:
            new_q = st.text_input("Add question", key="qa_new_q", label_visibility="collapsed")
        with c_add2:
            new_w = st.radio(
                "Weight",
                options=[1.0, 0.5, 0.0],
                index=0,
                format_func=lambda x: str(x),
                horizontal=True,
                key="qa_new_w",
                label_visibility="collapsed",
            )
        with c_add3:
            if st.button("➕", key="qa_add_btn") and (new_q or "").strip():
                qa_rows.append({"text": new_q.strip(), "weight": float(new_w)})
                # Request a clear on next run to avoid modifying a live widget value
                st.session_state["qa_add_clear"] = True
                st.rerun()

        # Scoring scheme info
        st.caption("Scoring scheme (fixed): Yes=1, Partial=0.5, No=0")
        total_weight = sum(float(r.get("weight", 1.0)) for r in qa_rows) if qa_rows else 0.0
        max_total = float(total_weight)
        suggested_cutoff = round(max_total * 0.65, 2) if max_total > 0 else 0.0
        existing = float(st.session_state.get("qa_cutoff", suggested_cutoff))
        default_cutoff = max(0.0, min(existing, max_total))
        st.session_state.setdefault("qa_cutoff", default_cutoff)
        st.session_state["qa_cutoff"] = st.number_input(
            "Cutoff minimum total score",
            min_value=0.0,
            max_value=max(0.0, max_total),
            value=default_cutoff,
            step=0.1,
        )

        if st.button("Save QA Checklist"):
            qac = {
                "answers": {"Yes": 1.0, "Partial": 0.5, "No": 0.0},
                "questions": qa_rows,
                "cutoff": float(st.session_state.get("qa_cutoff", 0.0)),
            }
            prot = _safe_read_json(protocol_path, {})
            prot.setdefault("topic", st.session_state.get("topic", ""))
            prot["qa_checklist"] = qac
            _safe_write_json(protocol_path, prot)
            st.success("QA checklist saved to reports/protocol.json")
            st.json({"qa_checklist": qac})
    else:
        st.caption("Click 'Generate AI checklist' or add your own questions.")

    # Step 7 — Define the Data Extraction form
    st.header("Define data extraction form")
    st.caption("Create a structured form (no AI) to capture study characteristics and findings.")

    cde1 = st.columns(1)[0]
    with cde1:
        create_form = st.button("Create data extraction form")

    # Helper to provide the requested well-structured default form (no AI)
    def _default_extraction_fields():
        return [
            {
                "name": "Research type",
                "type": "text",
                "help": "Theoretical (abstract, concepts) or Empirical (data/case studies).",
            },
            {
                "name": "By process phases, stages",
                "type": "text",
                "help": "Use lifecycle/framework stages (e.g., MAPE-K, context-aware lifecycle).",
            },
            {
                "name": "By technology, framework, or platform",
                "type": "text",
                "help": "Technologies, frameworks, tools or platforms used.",
            },
            {
                "name": "By application field and/or industry domain",
                "type": "text",
                "help": "Primary application field/domain (e.g., manufacturing, VR).",
            },
            {
                "name": "Gaps and challenges",
                "type": "text",
                "help": "Identified gaps/challenges and future needs.",
            },
            {
                "name": "Findings in research",
                "type": "text",
                "help": "Main findings (framework, algorithm, methodology, data model, approach).",
            },
            {
                "name": "Evaluation method",
                "type": "text",
                "help": "Case study, experiment, survey, mathematical demonstration, indicators.",
            },
        ]

    if create_form:
        st.session_state["de_fields"] = _default_extraction_fields()

    # Load saved form if available and session empty
    saved = _safe_read_json(protocol_path, {})
    saved_form = saved.get("extraction_form") or {}
    if saved_form and not st.session_state.get("de_fields"):
        st.session_state["de_fields"] = saved_form.get("fields", [])

    de_fields = st.session_state.get("de_fields") or []
    if de_fields:
        st.subheader("Fields")
        # Render each field line
        del_idx = []
        for i, f in enumerate(de_fields):
            c1, c2, c3, c4 = st.columns([0.05, 0.35, 0.22, 0.38])
            with c1:
                if st.button("✕", key=f"de_del_{i}"):
                    del_idx.append(i)
            with c2:
                f["name"] = st.text_input("Label", value=f.get("name", ""), key=f"de_name_{i}")
            with c3:
                f["type"] = st.selectbox(
                    "Type",
                    ["text", "long_text", "number", "boolean", "select", "multiselect"],
                    index=["text", "long_text", "number", "boolean", "select", "multiselect"].index(f.get("type", "text")),
                    key=f"de_type_{i}",
                )
            with c4:
                if f.get("type") in ("select", "multiselect"):
                    opts_str = ", ".join(f.get("options", []))
                    new_opts = st.text_input("Options (comma-separated)", value=opts_str, key=f"de_opts_{i}")
                    f["options"] = [s.strip() for s in new_opts.split(",") if s.strip()]
                else:
                    hint = st.text_input("Help (optional)", value=f.get("help", ""), key=f"de_help_{i}")
                    f["help"] = hint
        if del_idx:
            for idx in sorted(del_idx, reverse=True):
                del de_fields[idx]
            st.session_state["de_fields"] = de_fields

        # Add new field row
        ad1, ad2, ad3, ad4 = st.columns([0.35, 0.22, 0.35, 0.08])
        with ad1:
            nf_name = st.text_input("Add label", key="de_new_name")
        with ad2:
            nf_type = st.selectbox("Type", ["text", "long_text", "number", "boolean", "select", "multiselect"], index=0, key="de_new_type")
        with ad3:
            nf_opts = st.text_input("Options (for select)", key="de_new_opts") if nf_type in ("select", "multiselect") else st.text_input("Help (optional)", key="de_new_help")
        with ad4:
            if st.button("➕", key="de_add_btn") and (nf_name or "").strip():
                entry = {"name": nf_name.strip(), "type": nf_type}
                if nf_type in ("select", "multiselect"):
                    entry["options"] = [s.strip() for s in (nf_opts or "").split(",") if s.strip()]
                else:
                    entry["help"] = str(nf_opts or "").strip()
                de_fields.append(entry)
                st.session_state["de_fields"] = de_fields
                st.session_state["de_new_name"] = ""
                st.rerun()

        if st.button("Save Extraction Form"):
            form = {"fields": de_fields}
            prot = _safe_read_json(protocol_path, {})
            prot.setdefault("topic", st.session_state.get("topic", ""))
            prot["extraction_form"] = form
            _safe_write_json(protocol_path, prot)
            st.success("Data extraction form saved to reports/protocol.json")
            st.json(form)
    else:
        st.caption("Click 'Generate AI form' or add custom fields.")


# ---------- Conducting Phase ----------
def render_conducting():
    st.markdown("**Conducting Phase**")
    steps = [
        "1) Build digital library search strings",
        "2) Gather studies (arXiv)",
        "3) Refinement 1: Dedupe and screen",
        "4) Refinement 2: Assign quality scores",
        "5) Data extraction",
    ]
    st.write(" ")
    st.markdown(" · ".join(steps))

    # Step 1 — Build digital library search strings (requires PICOC)
    st.header("Build search strings")
    if st.session_state.get("picoc"):
        libs = st.session_state.get("libraries", ["arxiv"])  # default to arXiv if not chosen
        gen = st.button("Generate Search Strings", key="gen_search_conduct")
        if gen:
            st.session_state["base_query_text"] = build_base_query(st.session_state.get("picoc", {}))
            if "arxiv" in libs:
                st.session_state["arxiv_query_text"] = build_arxiv_query(st.session_state.get("picoc", {}))
            else:
                st.session_state.pop("arxiv_query_text", None)
        tab_labels = ["Base String"] + (["arXiv"] if "arxiv" in libs else [])
        tabs = st.tabs(tab_labels)
        with tabs[0]:
            st.text_area("", key="base_query_text", height=120, placeholder='("software") AND ("development") AND ("accuracy")')
        if "arxiv" in libs:
            with tabs[1]:
                current_base = st.session_state.get("base_query_text", "")
                snapshot = st.session_state.get("last_base_snapshot")
                if current_base and snapshot != current_base:
                    st.session_state["arxiv_query_text"] = convert_base_to_arxiv(current_base)
                    st.session_state["last_base_snapshot"] = current_base
                st.text_area("", key="arxiv_query_text", height=120, placeholder='(all:"software") AND (all:"development") AND (all:"accuracy")')
        if st.button("Save Search Strings", key="save_search_strings_conduct"):
            prot = _safe_read_json(protocol_path, {})
            base_value = (st.session_state.get("base_query_text") or "").strip()
            arxiv_value = (st.session_state.get("arxiv_query_text") or "").strip() if "arxiv" in libs else ""
            if "arxiv" in libs and (not arxiv_value and base_value):
                arxiv_value = convert_base_to_arxiv(base_value)
            prot["base_query"] = base_value
            prot["arxiv_query"] = arxiv_value
            _safe_write_json(protocol_path, prot)
            st.success("Search strings saved to reports/protocol.json")
            st.json({"base_query": prot.get("base_query"), "arxiv_query": prot.get("arxiv_query")})
    else:
        st.caption("Complete PICOC in Planning to enable this step.")

    # Step 2 — Gather studies (arXiv): requires saved/available arXiv query
    libs = st.session_state.get("libraries", ["arxiv"])  # default if none saved
    has_arxiv_query = ("arxiv" in libs) and bool(st.session_state.get("arxiv_query_text"))
    if has_arxiv_query:
        st.header("Gather studies (arXiv)")
        query = st.session_state.get("arxiv_query_text", "")
        st.code(query or "", language="text")

        def _fetch_arxiv(q: str, limit: int = 100):
            import arxiv
            # Use a larger page size to retrieve more per request, still guard empty pages
            client = arxiv.Client(page_size=min(50, limit), delay_seconds=2, num_retries=2)
            search = arxiv.Search(query=q, max_results=limit, sort_by=arxiv.SortCriterion.Relevance)
            results = []
            try:
                for r in client.results(search):
                    try:
                        results.append({
                            "entry_id": getattr(r, "entry_id", None),
                            "title": getattr(r, "title", ""),
                            "summary": getattr(r, "summary", ""),
                            "authors": [a.name for a in getattr(r, "authors", [])],
                            "published": r.published.strftime("%Y-%m-%d") if getattr(r, "published", None) else None,
                            "updated": r.updated.strftime("%Y-%m-%d") if getattr(r, "updated", None) else None,
                            "categories": list(getattr(r, "categories", [])),
                            "primary_category": getattr(r, "primary_category", None),
                            "journal_ref": getattr(r, "journal_ref", None),
                            "doi": getattr(r, "doi", None),
                            "abs_url": getattr(r, "entry_id", None),
                            "pdf_url": getattr(r, "pdf_url", None),
                            "links": [getattr(l, "href", None) for l in getattr(r, "links", [])],
                        })
                    except Exception:
                        continue
            except Exception as e:
                # The arxiv client may raise UnexpectedEmptyPageError on subsequent pages.
                # Return what we have so far rather than failing the run.
                st.warning(f"arXiv returned a partial page of results; showing {len(results)}.")
            return results

        def _relaxed_query_from_picoc() -> str:
            # If strict query returns 0, fall back to OR over Intervention + Population terms
            picoc = st.session_state.get("picoc", {}) or {}
            ints = picoc.get("Intervention", [])[:6]
            pops = picoc.get("Population", [])[:6]
            parts = []
            if ints:
                parts.append("(" + " OR ".join(f'all:"{t}"' for t in ints) + ")")
            if pops:
                parts.append("(" + " OR ".join(f'all:"{t}"' for t in pops) + ")")
            # Add CS category constraint
            if parts:
                parts.append("(" + " OR ".join(f"cat:{c}" for c in CS_CATEGORIES) + ")")
            else:
                parts = ["(" + " OR ".join(f"cat:{c}" for c in CS_CATEGORIES) + ")"]
            return " AND ".join(parts)

        # No special query builder here; we use the saved arXiv query from the Build step.

        if st.button("Fetch from arXiv", key="do_fetch_arXiv"):
            if not query.strip():
                st.error("arXiv query is empty. Generate Search Strings first.")
            else:
                with st.spinner("Fetching..."):
                    docs = _fetch_arxiv(query, limit=100)
                if not docs:
                    # Retry with a relaxed PICOC-derived query as a fallback only
                    relaxed = _relaxed_query_from_picoc()
                    if relaxed:
                        with st.spinner("No results with strict query. Retrying with a relaxed query..."):
                            docs = _fetch_arxiv(relaxed, limit=100)
                # Save raw results
                (reports_dir / "arxiv.json").write_text(json.dumps(docs, indent=2), encoding="utf-8")
                msg = f"Found {len(docs)} papers. Saved to reports/arxiv.json"
                if not docs:
                    msg += " (tip: adjust your PICOC or query)"
                st.success(msg)
                # Show that we used the arXiv query from the Build step
                st.caption("Used arXiv query from 'Build search strings':")
                st.code(query, language="text")
                # Preview with nicer columns (Title, Authors, Year, Journal, Links)
                import pandas as pd
                rows = []
                for d in docs:
                    year = (d.get("published") or "")[:4]
                    rows.append({
                        "Title": d.get("title", ""),
                        "Authors": ", ".join(d.get("authors", [])[:5]),
                        "Year": year,
                        "Journal": d.get("journal_ref") or "",
                        "PDF": d.get("pdf_url") or "",
                        "Link": d.get("abs_url") or "",
                    })
                if rows:
                    df = pd.DataFrame(rows)
                    # Start row numbering at 1 instead of 0
                    df.index = range(1, len(df) + 1)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        column_config={
                            "PDF": st.column_config.LinkColumn("PDF", display_text="pdf"),
                            "Link": st.column_config.LinkColumn("Link", display_text="abs"),
                        },
                    )

    # Protocol Export (enabled once all elements exist in this session)
    st.subheader("Protocol Export")
    libs = st.session_state.get("libraries", ["arxiv"])  # reevaluate
    session_ready = (
        st.session_state.get("topic")
        and st.session_state.get("picoc")
        and st.session_state.get("rqs")
        and st.session_state.get("base_query_text")
        and (st.session_state.get("arxiv_query_text") if "arxiv" in libs else True)
    )
    if session_ready:
        prot = _safe_read_json(protocol_path, {})
        prot.setdefault("topic", st.session_state.get("topic", ""))
        prot.setdefault("picoc", st.session_state.get("picoc", {}))
        prot.setdefault("rqs", st.session_state.get("rqs", []))
        prot.setdefault("base_query", st.session_state.get("base_query_text", ""))
        prot.setdefault("arxiv_query", st.session_state.get("arxiv_query_text", ""))
        if st.session_state.get("qa_questions"):
            prot.setdefault(
                "qa_checklist",
                {
                    "answers": {"Yes": 1.0, "Partial": 0.5, "No": 0.0},
                    "questions": st.session_state.get("qa_questions", []),
                    "cutoff": float(st.session_state.get("qa_cutoff", 0.0)),
                },
            )
        st.json(prot)
        st.download_button("Download protocol.json", data=json.dumps(prot, indent=2), file_name="protocol.json", mime="application/json")
    else:
        st.caption("Complete all Planning steps in this session to enable export.")

    # Step 3 — Refinement 1: Identify duplicates and evaluate inclusion/exclusion
    st.header("Step 3 — Refinement 1: Dedupe and screen")
    arxiv_path = reports_dir / "arxiv.json"
    if not arxiv_path.exists():
        st.info("Run 'Gather studies (arXiv)' first to fetch papers.")
    else:
        th = st.slider("Relevance threshold (semantic)", 0.0, 1.0, 0.4, 0.01)
        run_screen = st.button("Run screening")

        def _load_docs():
            try:
                data = json.loads(arxiv_path.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else []
            except Exception:
                return []

        def _year_in_range(published: str, fy: int | None, ty: int | None) -> bool:
            try:
                y = int((published or "")[:4])
                if fy and y < fy:
                    return False
                if ty and y > ty:
                    return False
                return True
            except Exception:
                return True

        def _dedupe_by_title(docs):
            seen = set()
            out = []
            for d in docs:
                t = (d.get("title") or "").lower()
                key = "".join(ch for ch in t if ch.isalnum())
                if key and key in seen:
                    continue
                seen.add(key)
                out.append(d)
            return out

        def _semantic_scores(texts, query):
            try:
                from sentence_transformers import SentenceTransformer, util
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                q_emb = model.encode([query], normalize_embeddings=True)
                d_emb = model.encode(texts, normalize_embeddings=True)
                sims = util.cos_sim(d_emb, q_emb).cpu().numpy().reshape(-1)
                return sims.tolist()
            except Exception:
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.metrics.pairwise import cosine_similarity
                    vec = TfidfVectorizer(stop_words="english", max_features=20000)
                    X = vec.fit_transform(texts + [query])
                    sims = cosine_similarity(X[:-1], X[-1])
                    import numpy as np
                    return np.asarray(sims).reshape(-1).tolist()
                except Exception:
                    return [0.0] * len(texts)

        if run_screen or "screen_df" not in st.session_state:
            docs = _dedupe_by_title(_load_docs())
            fy = st.session_state.get("period_from")
            ty = st.session_state.get("period_to")
            docs = [d for d in docs if _year_in_range(d.get("published"), fy, ty)]
            rqs = st.session_state.get("rqs", []) or []
            if rqs:
                query = " \n".join(rqs)
            else:
                p = st.session_state.get("picoc", {}) or {}
                query = " ".join([st.session_state.get("topic", ""), " ".join(sum((p.get(k, []) for k in p.keys()), []))])
            texts = [(d.get("title") or "") + "\n" + (d.get("summary") or "") for d in docs]
            sims = _semantic_scores(texts, query)
            try:
                import pandas as pd
            except Exception:
                pd = None
            rows = []
            for d, s in zip(docs, sims):
                year = (d.get("published") or "")[:4]
                rows.append({
                    "Include": bool(s >= th),
                    "Score": round(float(s), 3),
                    "Title": d.get("title", ""),
                    "Authors": ", ".join(d.get("authors", [])[:5]),
                    "Year": year,
                    "PDF": d.get("pdf_url") or "",
                    "Link": d.get("abs_url") or d.get("entry_id") or "",
                })
            if pd is not None:
                df = pd.DataFrame(rows)
                df.index = range(1, len(df) + 1)
                st.session_state["screen_df"] = df
            else:
                st.session_state["screen_df"] = None

        df = st.session_state.get("screen_df")
        if df is not None:
            st.subheader("Screening table")
            edited = st.data_editor(
                df,
                use_container_width=True,
                column_config={
                    "PDF": st.column_config.LinkColumn("PDF", display_text="pdf"),
                    "Link": st.column_config.LinkColumn("Link", display_text="abs"),
                },
                hide_index=False,
            )
            include_count = int(edited["Include"].sum())
            exclude_count = int(len(edited) - include_count)
            m1, m2 = st.columns(2)
            m1.metric("Include", include_count)
            m2.metric("Exclude", exclude_count)

            if st.button("Save screening decisions"):
                inc_list = edited[edited["Include"]].to_dict(orient="records")
                exc_list = edited[~edited["Include"]].to_dict(orient="records")
                out = {"include": inc_list, "exclude": exc_list}
                (reports_dir / "arxiv_screened.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
                st.success("Saved to reports/arxiv_screened.json")

    # Step 4 — Refinement 2: Assign quality scores (applies QA checklist to included papers)
    st.header("Step 4 — Refinement 2: Assign quality scores")
    screened_path = reports_dir / "arxiv_screened.json"
    qa = _safe_read_json(protocol_path, {}).get("qa_checklist") or {}
    if not screened_path.exists():
        st.info("Run 'Refinement 1: Dedupe and screen' and save decisions first.")
        return
    if not qa or not qa.get("questions"):
        st.info("Define and save a QA checklist in Planning → Step 6 before scoring.")
        return

    # Load included papers
    try:
        screened = json.loads(screened_path.read_text(encoding="utf-8"))
    except Exception:
        screened = {}
    included = screened.get("include") or []
    if not included:
        st.warning("No included papers to score. Include some in Step 3 first.")
        return

    # Prep QA model
    questions = qa.get("questions") or []
    answers_map = {"Yes": 1.0, "Partial": 0.5, "No": 0.0}
    weights = [float(max(0.0, min(5.0, q.get("weight", 1.0)))) for q in questions]
    texts = [str(q.get("text", "")).strip() for q in questions]
    max_total = float(sum(weights)) if weights else 0.0
    cutoff_saved = float(qa.get("cutoff", 0.0)) if qa.get("cutoff") is not None else (0.65 * max_total)
    # Allow user to tweak cutoff during scoring
    cutoff = st.slider(
        "Quality cutoff (minimum total score)",
        min_value=0.0,
        max_value=max_total if max_total > 0 else 0.0,
        value=min(cutoff_saved, max_total) if max_total > 0 else 0.0,
        step=0.1,
    )
    st.caption(f"Max total score = {max_total:.2f}")

    # Helper to sanitize keys (for session_state)
    def _key_from_paper(p: dict, idx: int) -> str:
        base = p.get("Link") or p.get("Title") or str(idx)
        return "qa2_" + "".join(ch if ch.isalnum() else "_" for ch in base)[:64] + f"_{idx}"

    def _qa_suggest_answers(p: dict, q_texts: list[str]) -> list[str]:
        """Very lightweight heuristic suggestions per question from title+abstract.
        Returns a list of 'Yes'/'Partial'/'No' for each question.
        """
        title = (p.get("Title") or p.get("title") or "").lower()
        abstr = (p.get("summary") or p.get("Summary") or "").lower()
        text = title + "\n" + abstr
        sugs: list[str] = []
        for qt in q_texts:
            ql = (qt or "").lower()
            ans = "Partial"
            if any(k in text for k in ["code available", "github", "reproduc", "implementation"]):
                if "reproduc" in ql or "implement" in ql:
                    ans = "Yes"
            if any(k in text for k in ["dataset", "data set", "benchmark", "hardware", "gpu", "cpu", "fpga", "hpc", "evaluation"]):
                if "experiment" in ql or "dataset" in ql or "hardware" in ql or "reported" in ql:
                    ans = "Yes"
            if any(k in text for k in ["metric", "accuracy", "precision", "recall", "f1", "auc", "baseline", "compare", "state-of-the-art", "sota"]):
                if "methodology" in ql or "metric" in ql or "appropriate" in ql:
                    ans = "Yes"
            if any(k in text for k in ["baseline", "compare", "compared", "benchmark"]):
                if "benchmarks" in ql or "prior" in ql or "validated" in ql:
                    ans = "Yes"
            if any(k in text for k in ["embedded", "edge", "iot", "hpc", "gpu", "fpga", "microcontroller"]):
                if "context" in ql or "hpc" in ql or "embedded" in ql:
                    ans = "Yes"
            sugs.append(ans)
        return sugs

    # Render per-paper scoring
    totals = []
    for idx, p in enumerate(included):
        title = p.get("Title") or p.get("title") or f"Paper {idx+1}"
        year = p.get("Year") or ""
        with st.expander(f"{idx+1}. {title} ({year})", expanded=(idx < 5)):
            # For each question render a radio and store in session_state
            row_scores = []
            # Seed suggested defaults once per paper
            suggestions = _qa_suggest_answers(p, texts) if texts else []
            for qi, (qt, w) in enumerate(zip(texts, weights)):
                opts = ["Yes", "Partial", "No"]
                key = f"{_key_from_paper(p, idx)}_q{qi}"
                # Default to suggestion -> stick in session_state before widget creation
                default_choice = suggestions[qi] if qi < len(suggestions) and suggestions[qi] in opts else "Partial"
                st.session_state.setdefault(key, default_choice)
                choice = st.radio(
                    qt or f"Q{qi+1}",
                    opts,
                    index=opts.index(st.session_state.get(key, default_choice)),
                    horizontal=True,
                    key=key,
                )
                row_scores.append(answers_map.get(choice, 0.5) * float(w))
            total = float(sum(row_scores))
            passed = bool(total >= cutoff)
            badge = "✅ Pass" if passed else "⚠️ Below cutoff"
            st.write(f"Score: {total:.2f} / {max_total:.2f} — {badge}")
            totals.append({"total": total, "pass": passed, "paper": p})

    # Summary and save
    passed_count = sum(1 for t in totals if t["pass"])
    failed_count = len(totals) - passed_count
    c1, c2 = st.columns(2)
    c1.metric("Pass", passed_count)
    c2.metric("Fail", failed_count)

    # Compact summary table + CSV export
    try:
        import pandas as pd
    except Exception:
        pd = None
    if pd is not None:
        rows = []
        for t in totals:
            p = t["paper"]
            rows.append({
                "Title": p.get("Title") or p.get("title", ""),
                "Year": p.get("Year") or (p.get("published", "")[:4] if p.get("published") else ""),
                "Score": round(float(t["total"]), 2),
                "Max": round(float(max_total), 2),
                "Pass": bool(t["pass"]),
                "PDF": p.get("PDF") or p.get("pdf_url") or "",
                "Link": p.get("Link") or p.get("abs_url") or p.get("entry_id") or "",
            })
        df_sum = pd.DataFrame(rows)
        if not df_sum.empty:
            df_sum.index = range(1, len(df_sum) + 1)
            st.subheader("Quality scores summary")
            st.dataframe(
                df_sum,
                use_container_width=True,
                column_config={
                    "PDF": st.column_config.LinkColumn("PDF", display_text="pdf"),
                    "Link": st.column_config.LinkColumn("Link", display_text="abs"),
                },
            )
            csv_data = df_sum.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download quality scores (CSV)",
                data=csv_data,
                file_name="arxiv_quality.csv",
                mime="text/csv",
            )

    if st.button("Save quality scores"):
        out_rows = []
        for idx, t in enumerate(totals):
            p = t["paper"]
            ans = {}
            for qi, (qt, w) in enumerate(zip(texts, weights)):
                key = f"{_key_from_paper(p, idx)}_q{qi}"
                ans_text = st.session_state.get(key, "Partial")
                ans_val = answers_map.get(ans_text, 0.5)
                ans[qt or f"Q{qi+1}"] = {"answer": ans_text, "value": ans_val, "weight": float(w)}
            out_rows.append({
                "paper": p,
                "score": float(t["total"]),
                "max": max_total,
                "pass": bool(t["pass"]),
                "cutoff": float(cutoff),
                "answers": ans,
            })
        _safe_write_json(reports_dir / "arxiv_quality.json", {"results": out_rows})
        # Also persist a CSV summary alongside JSON (best-effort)
        try:
            if pd is None:
                import pandas as pd  # type: ignore
            rows = []
            for t in totals:
                p = t["paper"]
                rows.append({
                    "Title": p.get("Title") or p.get("title", ""),
                    "Year": p.get("Year") or (p.get("published", "")[:4] if p.get("published") else ""),
                    "Score": round(float(t["total"]), 2),
                    "Max": round(float(max_total), 2),
                    "Pass": bool(t["pass"]),
                    "PDF": p.get("PDF") or p.get("pdf_url") or "",
                    "Link": p.get("Link") or p.get("abs_url") or p.get("entry_id") or "",
                })
            pd.DataFrame(rows).to_csv(reports_dir / "arxiv_quality.csv", index=False)
        except Exception:
            pass
        st.success("Saved to reports/arxiv_quality.json")

    # Step 5 — Data extraction (lightweight for taxonomy)
    st.header("Step 5 — Data extraction")
    qa_out_path = reports_dir / "arxiv_quality.json"
    extraction_form = _safe_read_json(protocol_path, {}).get("extraction_form", {}).get("fields", [])
    if not qa_out_path.exists():
        st.info("Run Step 4 and save quality scores first.")
        return
    if not extraction_form:
        st.info("Define and save a data extraction form in Planning → Step 7.")
        return

    # Load passed papers
    try:
        qa_out = json.loads(qa_out_path.read_text(encoding="utf-8"))
    except Exception:
        qa_out = {}
    passed = [r for r in (qa_out.get("results") or []) if r.get("pass")]
    if not passed:
        st.warning("No papers passed the QA cutoff. Adjust cutoff or include more.")
        return

    # New: one-click AI-assisted extraction
    def _download_pdf_text(url: str, max_pages: int = 6) -> str:
        try:
            import io
            import requests
            from pypdf import PdfReader
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=25)
            if resp.status_code != 200 or not resp.content:
                return ""
            data = io.BytesIO(resp.content)
            reader = PdfReader(data)
            pages = min(max_pages, len(reader.pages))
            text = []
            for i in range(pages):
                try:
                    text.append(reader.pages[i].extract_text() or "")
                except Exception:
                    continue
            return "\n".join(text)
        except Exception:
            return ""

    if st.button("Data extraction"):
        topic = st.session_state.get("topic", "")
        rqs = st.session_state.get("rqs", []) or []
        results = []
        for rec in passed:
            p = rec.get("paper", {})
            title = p.get("Title") or p.get("title") or ""
            abstract = p.get("summary") or p.get("Summary") or ""
            pdf_url = p.get("PDF") or p.get("pdf_url") or ""
            extra_text = _download_pdf_text(pdf_url) if pdf_url else ""
            try:
                raw = extract_data_for_paper(
                    topic=topic,
                    rqs=rqs,
                    form_fields=extraction_form,
                    title=title,
                    abstract=abstract,
                    extra_text=extra_text,
                )
                mapping = json.loads(raw)
                if not isinstance(mapping, dict):
                    mapping = {}
            except Exception:
                mapping = {}
            results.append({
                "paper": {
                    "Title": title,
                    "Year": p.get("Year") or (p.get("published", "")[:4] if p.get("published") else ""),
                    "PDF": pdf_url,
                    "Link": p.get("Link") or p.get("abs_url") or p.get("entry_id") or "",
                },
                "abstract": abstract,
                "fields": {str(f.get("name", "")).strip(): str(mapping.get(str(f.get("name", "")).strip(), "")) for f in extraction_form},
            })
        _safe_write_json(reports_dir / "extracted.json", results)
        st.success(f"Extracted {len(results)} papers to reports/extracted.json")
        try:
            import pandas as pd
            df = pd.DataFrame([
                {"Title": r["paper"]["Title"], **r.get("fields", {})} for r in results
            ])
            df.index = range(1, len(df) + 1)
            st.dataframe(df, use_container_width=True)
        except Exception:
            pass

    # Step 6 — Build the hierarchical taxonomy
    st.header("Step 6 — Build hierarchical taxonomy")
    extracted_path = reports_dir / "extracted.json"
    if not extracted_path.exists():
        st.info("Run 'Data extraction' first to create reports/extracted.json.")
        return

    # Prepare texts from extracted.json
    try:
        extracted = json.loads(extracted_path.read_text(encoding="utf-8"))
        docs = []
        titles = []
        for r in extracted:
            titles.append(r.get("paper", {}).get("Title", ""))
            text = (r.get("abstract") or "") + "\n" + "\n".join(str(v) for v in (r.get("fields") or {}).values() if v)
            docs.append(text.strip())
    except Exception:
        extracted, docs, titles = [], [], []

    if not docs:
        st.warning("No extracted data available.")
        return

    # Show engine availability (BERTopic vs fallback). Keep it minimal.
    engine_msg = ""
    try:
        from importlib.util import find_spec
        engine_msg = "BERTopic available (embeddings clustering)" if find_spec("bertopic") else "TF‑IDF fallback active (no BERTopic)"
    except Exception:
        engine_msg = "TF‑IDF fallback active (no BERTopic)"
    st.caption(f"Engine: {engine_msg}")

    if st.button("Build taxonomy"):
        taxonomy = {}
        used_bertopic = False
        # Prefer BERTopic (topics → parent clusters of topics)
        try:
            from bertopic import BERTopic  # type: ignore
            import plotly.io as pio  # type: ignore
            used_bertopic = True
            model = BERTopic(language="english", min_topic_size=3, calculate_probabilities=False, verbose=False)
            topics, _ = model.fit_transform(docs)
            info = model.get_topic_info().to_dict(orient="records")
            # Build topic texts and cluster topics into a few parent groups
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            topic_ids = [int(r["Topic"]) for r in info if int(r["Topic"]) != -1]
            topic_texts = []
            for tid in topic_ids:
                words = model.get_topic(tid) or []
                topic_texts.append(" ".join([w[0] if isinstance(w, (list, tuple)) else str(w) for w in words[:10]]))
            if not topic_texts:
                raise RuntimeError("No topics generated")
            vec = TfidfVectorizer(stop_words="english", max_features=3000)
            Xt = vec.fit_transform(topic_texts)
            k_par = max(2, min(6, len(topic_ids)//3 or 2))
            km_par = KMeans(n_clusters=k_par, n_init=10, random_state=42)
            par_labels = km_par.fit_predict(Xt)
            # label parents with top terms of their centroid
            parents = []
            terms = np.array(vec.get_feature_names_out())
            for pidx in range(k_par):
                idxs = np.where(par_labels == pidx)[0]
                if len(idxs) == 0:
                    continue
                centroid = km_par.cluster_centers_[pidx]
                top = terms[np.argsort(centroid)[-5:][::-1]].tolist()
                parents.append({"name": ", ".join(top), "children": []})
            # child topics under their parent
            assign = [None if t == -1 else int(t) for t in topics]
            doc_titles_by_topic = {}
            for i, t in enumerate(assign):
                if t is None:
                    continue
                doc_titles_by_topic.setdefault(int(t), []).append(titles[i] if i < len(titles) else f"Doc {i+1}")
            for t_idx, tid in enumerate(topic_ids):
                p = int(par_labels[t_idx])
                words = model.get_topic(tid) or []
                child_name = ", ".join([(w[0] if isinstance(w, (list, tuple)) else str(w)) for w in words[:3]]) or f"Topic {tid}"
                child = {"name": child_name, "docs": (doc_titles_by_topic.get(tid) or [])[:5]}
                parents[p]["children"].append(child)
            taxonomy = {"engine": "bertopic_h2", "root": (st.session_state.get("topic") or "Root"), "parents": parents}
        except Exception as e:
            # Fallback: two-level TF‑IDF KMeans on documents
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.cluster import KMeans
                import numpy as np
                vec = TfidfVectorizer(stop_words="english", max_features=5000)
                X = vec.fit_transform(docs)
                terms = np.array(vec.get_feature_names_out())
                k_par = max(2, min(6, len(docs)//3 or 2))
                km_par = KMeans(n_clusters=k_par, n_init=10, random_state=42)
                par_labels = km_par.fit_predict(X)
                parents = []
                for pidx in range(k_par):
                    idxs = np.where(par_labels == pidx)[0]
                    if len(idxs) == 0:
                        continue
                    centroid = km_par.cluster_centers_[pidx]
                    top = terms[np.argsort(centroid)[-5:][::-1]].tolist()
                    # child clustering within parent
                    subX = X[idxs]
                    k_child = max(2, min(6, subX.shape[0]//3 or 2))
                    km_child = KMeans(n_clusters=k_child, n_init=10, random_state=42)
                    child_labels = km_child.fit_predict(subX)
                    children = []
                    for cidx in range(k_child):
                        csel = np.where(child_labels == cidx)[0]
                        if len(csel) == 0:
                            continue
                        ccent = km_child.cluster_centers_[cidx]
                        ctop = terms[np.argsort(ccent)[-3:][::-1]].tolist()
                        docs_in_child = [titles[idxs[j]] if idxs[j] < len(titles) else f"Doc {int(idxs[j])+1}" for j in csel[:5]]
                        children.append({"name": ", ".join(ctop), "docs": docs_in_child})
                    parents.append({"name": ", ".join(top), "children": children})
                taxonomy = {"engine": "tfidf_kmeans_h2", "root": (st.session_state.get("topic") or "Root"), "parents": parents}
            except Exception as ee:
                st.error(f"Fallback clustering failed: {ee}")

        # Save taxonomy data regardless of engine
        if taxonomy:
            _safe_write_json(reports_dir / "taxonomy.json", taxonomy)
            st.success("Saved taxonomy to reports/taxonomy.json")
            # HTML tree: root -> parents -> children (with example docs)
            def _tree_html_h2(t):
                root = (t.get("root") or st.session_state.get("topic") or "Root").strip() or "Root"
                parts = ["<html><body><div style='font-family:system-ui;'><h3>Hierarchy</h3>"]
                parts.append(f"<ul><li><strong>{root}</strong><ul>")
                for p in t.get("parents", []):
                    parts.append(f"<li>{p.get('name','Cluster')}<ul>")
                    for ch in p.get("children", []):
                        docs_li = ''.join([f"<li style='color:#777'>{d}</li>" for d in ch.get("docs", [])])
                        parts.append(f"<li>{ch.get('name','Topic')}<ul>{docs_li}</ul></li>")
                    parts.append("</ul></li>")
                parts.append("</ul></li></ul></div></body></html>")
                return "\n".join(parts)

            html_tree = _tree_html_h2(taxonomy)
            (reports_dir / "taxonomy_hierarchy.html").write_text(html_tree, encoding="utf-8")
            try:
                import streamlit.components.v1 as components
                components.html(html_tree, height=600, scrolling=True)
            except Exception:
                st.info("Saved hierarchy to reports/taxonomy_hierarchy.html")

    # (Old manual extraction UI removed.)


# ---------- Page Shell ----------
st.title("Systematic Literature Review Automation")
st.caption("AI‑assisted planning and conducting phases for CS SLRs.")

tab_planning, tab_conducting = st.tabs(["Planning", "Conducting"])
with tab_planning:
    render_planning()
with tab_conducting:
    render_conducting()
