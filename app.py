import os
import json
import uuid
import re
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import streamlit as st
try:
    import requests  # type: ignore
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False
try:
    import json5  # type: ignore
    HAS_JSON5 = True
except Exception:
    HAS_JSON5 = False


DB_CSV_PATH = os.path.join(os.path.dirname(__file__), "agents_db.csv")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# SystemPromptBuilderAgent: specialized agent to craft a robust, structured system prompt
SYSTEM_PROMPT_BUILDER = (
    "**ROLE**: You are the \"Prompt Architect,\" a world-class expert in designing and refining System Prompts for AI agents. "
    "Your goal is to maximize the reliability and performance of the target agent.\n\n"
    "**GOAL**: Convert the user's high-level description of a target agent's purpose into a robust, structured System Prompt template "
    "suitable for direct use in an agent framework.\n\n"
    "**INPUT**: The user will provide a single, brief string describing the target agent's core function (e.g., \"This agent must write python code for a given requirement\").\n\n"
    "**PROCESS INSTRUCTIONS**:\n"
    "1.  **Infer Role & Persona**: Based on the input, infer the most effective specialized role (e.g., Expert Test Engineer, Data Transformation Utility, Rigor-Focused QA Analyst).\n"
    "2.  **Define Goal**: Use the user's input as the core objective for the target agent's GOAL section.\n"
    "3.  **Set Constraints**: Always include strict operational constraints (e.g., no casual chat, strictly follow output format, use tools).\n"
    "4.  **Suggest Tools**: If the user's purpose implies external knowledge (e.g., \"latest news,\" \"research,\" \"search\"), suggest the inclusion of a tool like `Google Search` in the prompt's constraints.\n"
    "5.  **Output Format**: Explicitly instruct the target agent on how its final response must be formatted (e.g., always use code blocks, must adhere to a schema).\n\n"
    "**OUTPUT FORMAT (MANDATORY)**:\n"
    "Your output must be the complete, structured System Prompt for the *target* agent. You must include the following sections, formatted using bold headings:\n\n"
    "ROLE: [Your inferred, specialized role for the target agent.]\n\n"
    "GOAL: [The primary objective, based on the user's initial input.]\n\n"
    "CONSTRAINTS:\n\n"
    "[Constraint 1: Be concise, accurate, and only use specified tools.]\n\n"
    "[Constraint 2: If input is ambiguous, request clarification before proceeding.]\n\n"
    "[Constraint 3: Adhere strictly to best practices for the task (e.g., linting code, proper markdown).]\n\n"
    "[Constraint 4: Do NOT engage in conversation outside of this defined role.]\n\n"
    "OUTPUT FORMAT:\n"
    "[Clear instructions on the exact format of the final response, e.g., 'Return only Python code in a single markdown block.']\n"
)


def slugify(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-"
    v = value.strip().lower()
    out = []
    prev_dash = False
    for ch in v:
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    slug = "".join(out).strip("-")
    return slug or f"agent-{uuid.uuid4().hex[:6]}"


AGENT_TYPES = [
    "General Purpose",
    "Quality Engineering",
    "Unit Testing",
    "Utility: Search",
    "Utility: File",
    "Utility: Image",
]

COMMON_TAGS = [
    "python",
    "pytest",
    "security",
    "performance",
    "accessibility",
    "ci/cd",
    "refactor",
    "static-analysis",
]


def suggest_agent_config(purpose: str) -> Dict:
    """First try local Ollama for JSON suggestions; fall back to heuristics."""
    # Attempt Ollama call if requests is available
    if HAS_REQUESTS:
        try:
            cfg = _ollama_suggest_config(purpose)
            if cfg:
                # Mark source for UI feedback
                st.session_state["agent_config_source"] = "ollama"
                return cfg
        except Exception as e:
            # Show a subtle message once per session
            st.session_state["agent_config_source_error"] = str(e)
            # Continue to heuristic fallback
            pass

    st.session_state["agent_config_source"] = "heuristic"
    p = purpose.lower()
    # Heuristic type/tags
    if any(k in p for k in ["unit test", "unit-test", "pytest", "jest", "test suite", "testing"]):
        a_type = "Unit Testing"
        tags = ["unit-test", "pytest" if "python" in p else "jest", "edge-cases"]
        name = "Unit Test Generator"
        description = "Generates focused unit test suites based on code and requirements."
        persona = "Thorough, edge-case-focused test author"
    elif any(k in p for k in ["quality", "bug", "lint", "security", "accessibility", "review"]):
        a_type = "Quality Engineering"
        tags = ["static-analysis", "security" if "security" in p else "quality", "review"]
        name = "Quality Bug Detector"
        description = "Analyzes code for quality issues and improvement opportunities."
        persona = "Rigorous but friendly QA reviewer"
    else:
        a_type = "General Purpose"
        tags = ["general", "assistant"]
        name = "General Purpose Assistant"
        description = "Helpful agent for broad utility tasks given a clear purpose."
        persona = "Helpful, concise assistant"

    system_prompt = (
        "You are an AI agent. Understand the user's goal and produce precise, actionable output. "
        "Ask clarifying questions when necessary. Be concise, correct, and cite assumptions."
    )

    # Default model config (simulated local)
    model_name = "local-llm-sim"
    temperature = 0.4
    max_tokens = 2048

    return {
        "name": name,
        "description": description,
        "system_prompt": system_prompt,
        "type": a_type,
        "tags": tags,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "persona": persona,
    }


def _extract_json_block(text: str) -> str:
    """Extract first top-level JSON object from a text blob."""
    if not text:
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _normalize_suggestion(d: Dict, purpose: str) -> Dict:
    # Defaults
    model_name = str(d.get("model_name") or OLLAMA_MODEL)
    try:
        temperature = float(d.get("temperature", 0.4))
    except Exception:
        temperature = 0.4
    temperature = max(0.0, min(1.0, temperature))
    try:
        max_tokens = int(d.get("max_tokens", 2048))
    except Exception:
        max_tokens = 2048

    a_type = str(d.get("type") or "General Purpose")
    if a_type not in AGENT_TYPES:
        a_type = "General Purpose"

    tags = d.get("tags", [])
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = [t.strip() for t in tags.split(",") if t.strip()]
    if not isinstance(tags, list):
        tags = []
    tags = [str(t) for t in tags]

    name = str(d.get("name") or "Generated Agent")
    description = str(d.get("description") or f"Agent generated for purpose: {purpose[:120]}")
    system_prompt = str(
        d.get("system_prompt")
        or (
            "You are an AI agent. Understand the user's goal and produce precise, actionable output. "
            "Ask clarifying questions when necessary. Be concise and correct."
        )
    )
    persona = str(d.get("persona") or "Helpful, concise assistant")

    return {
        "name": name,
        "description": description,
        "system_prompt": system_prompt,
        "type": a_type,
        "tags": tags,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "persona": persona,
    }


def _ollama_suggest_config(purpose: str) -> Dict:
    """Call local Ollama to get a JSON config suggestion for the given purpose using the SystemPromptBuilderAgent."""
    # We instruct the model to return ONLY JSON and include a full structured system_prompt string per the template.
    prompt = (
        SYSTEM_PROMPT_BUILDER
        + "\n\nNow, based on the INPUT below, produce a JSON object with these keys only: "
        + "name (string), description (string), type (string chosen from "
        + str(AGENT_TYPES)
        + "), tags (array of strings), persona (string), system_prompt (string containing the complete structured prompt as specified above).\n"
        + "Return STRICT JSON and no commentary.\n\n"
        + f"INPUT: {purpose}\n"
    )

    # Prefer chat API if available; fallback to generate
    url_generate = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {"model": get_ollama_model(), "prompt": prompt, "stream": False, "format": "json"}

    resp = requests.post(url_generate, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response") or data.get("text") or ""
    st.session_state["ollama_raw_output"] = text
    json_text = _extract_json_block(text)
    parsed = _try_parse_json_relaxed(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("Ollama response is not a JSON object")
    return _normalize_suggestion(parsed, purpose)


def _try_parse_json_relaxed(text: str):
    # First: strict JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # Second: json5 if available
    if HAS_JSON5:
        try:
            return json5.loads(text)
        except Exception:
            pass
    # Third: light repairs for common issues
    fixed = text
    # Replace smart quotes with normal quotes
    fixed = fixed.replace("“", '"').replace("”", '"').replace("’", "'")
    # Quote unquoted keys: { key: value } -> { "key": value }
    try:
        fixed = re.sub(r'([,{]\s*)([A-Za-z_][A-Za-z0-9_ \-]*)(\s*:) ', r'\1"\2"\3 ', fixed)
    except Exception:
        pass
    # Convert single-quoted strings to double quotes where plausible
    try:
        fixed = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", lambda m: '"' + m.group(1).replace('"', '\\"') + '"', fixed)
    except Exception:
        pass
    # Remove trailing commas before } or ]
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    except Exception:
        pass
    # Try strict then json5 again
    try:
        return json.loads(fixed)
    except Exception:
        if HAS_JSON5:
            try:
                return json5.loads(fixed)
            except Exception:
                pass
    return None


def get_ollama_model() -> str:
    # Allow runtime selection via sidebar; fallback to env default
    return st.session_state.get("ollama_model", OLLAMA_MODEL)


def set_ollama_model(model: str) -> None:
    st.session_state["ollama_model"] = model


def fetch_ollama_models() -> List[str]:
    if not HAS_REQUESTS:
        return []
    try:
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
        models = []
        for item in data.get("models", []) or data.get("tags", []):
            # Ollama returns objects with 'name' and maybe 'model' fields
            name = item.get("name") or item.get("model")
            if name:
                models.append(str(name))
        return sorted(list(set(models)))
    except Exception:
        return []


def render_sidebar_models() -> None:
    with st.sidebar:
        with st.expander("Ollama Models", expanded=False):
            models = fetch_ollama_models()
            if not models:
                st.write("No models found or Ollama unreachable.")
                st.caption(f"Base URL: {OLLAMA_BASE_URL}")
                st.caption("Ensure Ollama is running: https://ollama.com/")
            else:
                # Preselect current or default
                current = get_ollama_model()
                if current not in models:
                    current = models[0]
                    set_ollama_model(current)
                selected = st.selectbox("Default model", options=models, index=models.index(current), key="ollama_model_select")
                if selected != get_ollama_model():
                    set_ollama_model(selected)


def get_current_user() -> str:
    for key in ("AGENTDB_USER", "GIT_AUTHOR_NAME", "USERNAME", "USER"):
        v = os.getenv(key)
        if v:
            return v
    try:
        return os.getlogin()
    except Exception:
        return "local_user"


CSV_COLUMNS = [
    # System Generated
    "agent_id",
    "status",
    "version",
    "created_at",
    "created_by",
    "updated_at",
    "updated_by",
    # Suggested & Editable
    "name",
    "description",
    "system_prompt",
    "type",
    "tags",  # stored as JSON list
    # User Configurable (non-LLM params)
    "persona",
]


def load_db() -> pd.DataFrame:
    if os.path.exists(DB_CSV_PATH):
        try:
            df = pd.read_csv(DB_CSV_PATH, dtype=str)
        except Exception:
            df = pd.DataFrame(columns=CSV_COLUMNS)
    else:
        df = pd.DataFrame(columns=CSV_COLUMNS)

    # Ensure all expected columns exist and types for numeric fields are reasonable when used later
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def save_db(df: pd.DataFrame) -> None:
    # Respect column order
    out = df.copy()
    out = out.reindex(columns=CSV_COLUMNS)
    out.to_csv(DB_CSV_PATH, index=False)


def append_record_to_csv(record: Dict) -> None:
    df = load_db()
    df = pd.concat([df, pd.DataFrame([record], columns=CSV_COLUMNS)], ignore_index=True)
    save_db(df)


def ensure_tags_serialized(tags: List[str]) -> str:
    try:
        return json.dumps(tags, ensure_ascii=False)
    except Exception:
        # Fallback to comma separated
        return ",".join(tags)


def parse_tags(value: str) -> List[str]:
    if value is None or str(value).strip() == "":
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    # fallback: comma-separated
    return [t.strip() for t in str(value).split(",") if t.strip()]


def render_create_tab():
    st.header("Create Agent")
    st.caption("Use the purpose-driven flow to generate a draft, then refine and save.")

    # Step 1: Define Purpose
    st.subheader("Step 1: Define Purpose")
    purpose = st.text_area(
        "Describe the agent's purpose",
        placeholder=(
            "e.g., This agent must take Python code as input and generate a Pytest "
            "unit test suite for it, focusing on edge cases."
        ),
        key="purpose_input",
        height=120,
    )

    cols = st.columns([1, 1, 2])
    with cols[0]:
        suggest_clicked = st.button("Suggest Agent Configuration", type="primary", key="suggest_btn")
    with cols[1]:
        reset_clicked = st.button("Reset", key="reset_btn")

    if reset_clicked:
        st.session_state.pop("agent_config", None)
        st.experimental_rerun()

    if suggest_clicked:
        if not purpose or not purpose.strip():
            st.warning("Please enter a purpose before requesting suggestions.")
        else:
            with st.spinner("Generating suggestions (simulated)..."):
                st.session_state["agent_config"] = suggest_agent_config(purpose)
                source = st.session_state.get("agent_config_source", "heuristic")
                if source == "ollama":
                    st.success(f"Configuration suggested by Ollama ({get_ollama_model()}).")
                else:
                    err = st.session_state.get("agent_config_source_error")
                    if err:
                        st.info(f"Falling back to heuristic suggestions (Ollama error: {err}).")
                    else:
                        st.info("Heuristic suggestions generated.")

    # Step 2: Review, Refine, and Save
    st.subheader("Step 2: Review, Refine, and Save")
    cfg = st.session_state.get("agent_config")

    if cfg is None:
        st.info("Enter a purpose above and click 'Suggest Agent Configuration' to continue.")
        return

    with st.form("agent_form", clear_on_submit=False):
        st.markdown("### Core Details")
        name = st.text_input("Agent Name", value=cfg.get("name", ""), max_chars=100)
        description = st.text_area("Description", value=cfg.get("description", ""), height=100)
        system_prompt = st.text_area("System Prompt", value=cfg.get("system_prompt", ""), height=180)

        a_type = st.selectbox("Type", options=AGENT_TYPES, index=max(AGENT_TYPES.index(cfg.get("type", AGENT_TYPES[0])), 0))

        # Tags editor: multiselect + free-text additions
        existing_tag_options = sorted(list(set(COMMON_TAGS + cfg.get("tags", []))))
        tags_sel = st.multiselect("Tags", options=existing_tag_options, default=cfg.get("tags", []))
        extra_tags_text = st.text_input("Extra Tags (comma-separated)", value="")
        persona = st.text_input("Persona", value=cfg.get("persona", ""))

        submitted = st.form_submit_button("Save Agent to Database", use_container_width=True)

    if submitted:
        # Merge tags
        extra = [t.strip() for t in extra_tags_text.split(",") if t.strip()]
        tags = list(dict.fromkeys([*tags_sel, *extra]))  # unique while preserving order

        # Generate system fields
        now = datetime.now(timezone.utc).isoformat()
        created_by = get_current_user()
        slug = slugify(name or "agent")
        agent_id = f"{slug}-{uuid.uuid4().hex[:6]}"

        record = {
            "agent_id": agent_id,
            "status": "Active",
            "version": "1.0.0",
            "created_at": now,
            "created_by": created_by,
            "updated_at": now,
            "updated_by": created_by,
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "type": a_type,
            "tags": ensure_tags_serialized(tags),
            "persona": persona,
        }

        try:
            append_record_to_csv(record)
            st.success(f"Agent saved to {os.path.basename(DB_CSV_PATH)} with id {agent_id}")
            with st.expander("Saved Record", expanded=False):
                st.json(record)
        except Exception as e:
            st.error(f"Failed to save agent: {e}")


def render_browse_edit_tab():
    st.header("Browse / Edit / Delete Agents")
    df = load_db()

    if df.empty:
        st.info("No agents found yet. Create one in the Create tab.")
        return

    # Friendly view for table
    view_df = df.copy()
    view_df["tags"] = view_df["tags"].apply(lambda v: ", ".join(parse_tags(v)))

    with st.expander("Filters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            search = st.text_input("Search (name/description/tags)", value="")
            type_opts = sorted([t for t in AGENT_TYPES if t in set(df["type"].dropna().tolist())])
            type_sel = st.multiselect("Type", options=type_opts, default=[])
        with col2:
            status_opts = sorted([s for s in ["Active", "Draft", "Deprecated", "Maintenance"] if s in set(df["status"].dropna().tolist())])
            status_sel = st.multiselect("Status", options=status_opts, default=[])

    filtered = view_df
    if search.strip():
        q = search.strip().lower()
        filtered = filtered[
            filtered.apply(
                lambda r: any(
                    q in str(r[c]).lower()
                    for c in ["name", "description", "tags", "system_prompt"]
                ),
                axis=1,
            )
        ]
    if type_sel:
        filtered = filtered[filtered["type"].isin(type_sel)]
    if status_sel:
        filtered = filtered[filtered["status"].isin(status_sel)]

    st.dataframe(
        filtered[
            [
                "agent_id",
                "name",
                "type",
                "status",
                "version",
                "tags",
                "updated_at",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    if filtered.empty:
        st.warning("No agents match the current filters.")
        return

    # Selection
    options = [f"{r.agent_id} — {r.name}" for _, r in filtered.iterrows()]
    selected_display = st.selectbox("Select agent to edit/delete", options=options, key="select_agent")
    selected_id = selected_display.split(" — ", 1)[0]

    # Extract original row from df (not view_df) to retain raw values
    row = df[df["agent_id"] == selected_id].iloc[0]

    st.markdown("### Selected Agent")
    st.caption(f"Agent ID: {row['agent_id']}")

    with st.expander("Edit Agent", expanded=True):
        with st.form("edit_form", clear_on_submit=False):
            st.markdown("#### Core Details")
            name = st.text_input("Agent Name", value=row.get("name", ""))
            description = st.text_area("Description", value=row.get("description", ""), height=100)
            system_prompt = st.text_area("System Prompt", value=row.get("system_prompt", ""), height=160)
            a_type = st.selectbox(
                "Type", options=AGENT_TYPES, index=max(AGENT_TYPES.index(row.get("type", AGENT_TYPES[0])), 0)
            )

            # status and version editable
            col_s, col_v = st.columns(2)
            with col_s:
                status = st.selectbox("Status", options=["Active", "Draft", "Deprecated", "Maintenance"], index=["Active", "Draft", "Deprecated", "Maintenance"].index(row.get("status", "Active")))
            with col_v:
                version = st.text_input("Version", value=row.get("version", "1.0.0"))

            # Tags editor
            existing_tag_options = sorted(list(set(COMMON_TAGS + parse_tags(row.get("tags", "")))))
            current_tags = parse_tags(row.get("tags", ""))
            tags_sel = st.multiselect("Tags", options=existing_tag_options, default=current_tags)
            extra_tags_text = st.text_input("Extra Tags (comma-separated)", value="")
            persona = st.text_input("Persona", value=row.get("persona", ""))

            update_clicked = st.form_submit_button("Update Agent", use_container_width=True)

        if update_clicked:
            try:
                extra = [t.strip() for t in extra_tags_text.split(",") if t.strip()]
                tags = list(dict.fromkeys([*tags_sel, *extra]))

                now = datetime.now(timezone.utc).isoformat()
                updated_by = get_current_user()

                idx = df.index[df["agent_id"] == selected_id][0]
                df.at[idx, "name"] = name
                df.at[idx, "description"] = description
                df.at[idx, "system_prompt"] = system_prompt
                df.at[idx, "type"] = a_type
                df.at[idx, "status"] = status
                df.at[idx, "version"] = version
                df.at[idx, "tags"] = ensure_tags_serialized(tags)
                df.at[idx, "persona"] = persona
                df.at[idx, "updated_at"] = now
                df.at[idx, "updated_by"] = updated_by

                save_db(df)
                st.success("Agent updated.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to update agent: {e}")

    st.markdown("### Danger Zone")
    del_cols = st.columns([1, 2])
    with del_cols[0]:
        confirm = st.checkbox("Confirm delete", key="confirm_delete")
    with del_cols[1]:
        if st.button("Delete Agent", type="secondary", disabled=not confirm):
            try:
                new_df = df[df["agent_id"] != selected_id].copy()
                save_db(new_df)
                st.success("Agent deleted.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to delete agent: {e}")


def render_system_prompt_builder_tab():
    st.header("System Prompt Builder Agent")
    st.caption("Generate a robust, structured system prompt and core details from a simple requirement. No data is saved.")

    requirement = st.text_area(
        "User Requirement",
        placeholder="e.g., This agent must write python code for a given requirement",
        height=120,
        key="spb_requirement",
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        run_clicked = st.button("Generate System Prompt", type="primary", key="spb_generate")
    with col2:
        st.caption(f"Model: {get_ollama_model()}  •  Base URL: {OLLAMA_BASE_URL}")

    if run_clicked:
        if not requirement or not requirement.strip():
            st.warning("Please enter a requirement to generate a system prompt.")
            return
        with st.spinner("Generating (via Ollama or fallback)..."):
            cfg = suggest_agent_config(requirement)
            source = st.session_state.get("agent_config_source", "heuristic")
            if source == "ollama":
                st.success(f"Generated by Ollama ({get_ollama_model()}).")
            else:
                err = st.session_state.get("agent_config_source_error")
                if err:
                    st.info(f"Falling back to heuristic (Ollama error: {err}).")
                else:
                    st.info("Heuristic suggestion generated.")

        # Display core details
        st.markdown("### Core Agent Details")
        dcols = st.columns(2)
        with dcols[0]:
            st.text_input("Proposed Name", value=cfg.get("name", ""), key="spb_name", disabled=True)
            st.text_input("Type", value=cfg.get("type", ""), key="spb_type", disabled=True)
            st.text_input("Persona", value=cfg.get("persona", ""), key="spb_persona", disabled=True)
        with dcols[1]:
            tags_str = ", ".join([str(t) for t in cfg.get("tags", [])])
            st.text_input("Tags", value=tags_str, key="spb_tags", disabled=True)
            st.text_area("Description", value=cfg.get("description", ""), key="spb_desc", height=100, disabled=True)

        st.markdown("### Structured System Prompt")
        system_prompt_val = str(cfg.get("system_prompt", ""))
        st.text_area("System Prompt", value=system_prompt_val, height=300, key="spb_system_prompt", disabled=True)

        # Download options
        st.markdown("### Export")
        export_obj = {
            "name": cfg.get("name"),
            "description": cfg.get("description"),
            "type": cfg.get("type"),
            "tags": cfg.get("tags", []),
            "persona": cfg.get("persona"),
            "system_prompt": system_prompt_val,
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_obj, ensure_ascii=False, indent=2),
            file_name="system_prompt_config.json",
            mime="application/json",
        )
        st.download_button(
            label="Download Prompt (.md)",
            data=system_prompt_val,
            file_name="system_prompt.md",
            mime="text/markdown",
        )


def main():
    st.set_page_config(page_title="Agent Database Builder", page_icon="🤖", layout="wide")
    st.title("Agent Database Builder")
    st.caption("Create, browse, edit, and delete AI agents aligned with your schema. Stores to agents_db.csv")

    # Sidebar: Ollama models list and selector
    render_sidebar_models()

    tabs = st.tabs(["System Prompt Builder", "Create", "Browse / Edit / Delete"])
    with tabs[0]:
        render_system_prompt_builder_tab()
    with tabs[1]:
        render_create_tab()
    with tabs[2]:
        render_browse_edit_tab()


if __name__ == "__main__":
    main()
