from __future__ import annotations

import html
import os
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def get_api_base_url() -> str:
    """Read the API URL from Streamlit secrets first, then environment variables."""
    secret_url = ""
    try:
        secret_url = str(st.secrets.get("API_BASE_URL", "")).strip()
    except Exception:
        # st.secrets is optional for local runs.
        pass

    return (
        secret_url
        or os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    ).rstrip("/")


st.set_page_config(
    page_title="NOVA | Customer Support Intelligence",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = get_api_base_url()
API_DOCS_URL = f"{API_BASE_URL}/docs"


# -----------------------------------------------------------------------------
# Visual system
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

      :root {
        --nova-ink: #172238;
        --nova-muted: #6b7890;
        --nova-blue: #4263eb;
        --nova-blue-dark: #2f4ac5;
        --nova-teal: #0f9f91;
        --nova-line: #e6eaf2;
        --nova-canvas: #f5f7fb;
        --nova-card: #ffffff;
      }

      html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
      }
      .stApp { background: var(--nova-canvas); color: var(--nova-ink); }
      [data-testid="stHeader"] { background: rgba(245,247,251,.88); }
      [data-testid="stToolbar"] { right: 1rem; }
      .block-container { max-width: 1450px; padding-top: 1.4rem; padding-bottom: 3rem; }
      #MainMenu, footer { visibility: hidden; }

      h1, h2, h3, h4 { font-family: 'Manrope', sans-serif !important; color: var(--nova-ink); letter-spacing: -.025em; }
      h1 { font-size: clamp(1.7rem, 2.2vw, 2.25rem) !important; font-weight: 800 !important; }
      h2 { font-size: 1.45rem !important; font-weight: 750 !important; }
      h3 { font-size: 1.12rem !important; font-weight: 750 !important; }
      p, label, li { color: #475569; }
      [data-testid="stCaptionContainer"] p { color: var(--nova-muted); }

      [data-testid="stSidebar"] {
        background: #111a2e;
        border-right: 1px solid rgba(255,255,255,.07);
      }
      [data-testid="stSidebar"] * { color: #e9eefb; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aebad0; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: .45rem .6rem; border-radius: .65rem; margin-bottom: .15rem;
      }
      [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.07); }
      [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12); }
      [data-testid="stSidebar"] a { color: #9db4ff !important; }

      [data-testid="stMetric"] {
        background: var(--nova-card); border: 1px solid var(--nova-line);
        border-radius: 15px; padding: 1rem 1.1rem; box-shadow: 0 3px 12px rgba(26,39,72,.025);
      }
      [data-testid="stMetricLabel"] p { color: #697790 !important; font-size: .83rem; font-weight: 600; }
      [data-testid="stMetricValue"] { color: var(--nova-ink); font-family: 'Manrope', sans-serif; font-weight: 800; }

      [data-testid="stForm"] {
        background: #fff; border: 1px solid var(--nova-line); border-radius: 16px;
        padding: 1.15rem 1.2rem 1rem; box-shadow: 0 4px 18px rgba(26,39,72,.025);
      }
      [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
      [data-testid="stSelectbox"] [data-baseweb="select"],
      [data-testid="stMultiSelect"] [data-baseweb="select"] {
        border-radius: 10px !important;
      }
      [data-testid="stTextArea"] textarea { line-height: 1.55; }
      .stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
        border-radius: 10px; font-weight: 700; min-height: 2.65rem;
        transition: transform .12s ease, box-shadow .12s ease;
      }
      .stButton button:hover, .stDownloadButton button:hover,
      [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px); box-shadow: 0 6px 16px rgba(39,68,160,.12);
      }
      button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
        background: var(--nova-blue); border-color: var(--nova-blue); color: #fff;
      }
      [data-testid="stDataFrame"] { border: 1px solid var(--nova-line); border-radius: 12px; overflow: hidden; }
      [data-testid="stExpander"] { background: #fff; border: 1px solid var(--nova-line); border-radius: 12px; }
      [data-testid="stAlert"] { border-radius: 12px; }

      .nova-brand { display:flex; align-items:center; gap:.75rem; padding:.25rem 0 1.15rem; }
      .nova-logo { width:43px; height:43px; border-radius:14px; display:flex; align-items:center; justify-content:center;
        background: linear-gradient(140deg,#6484ff,#4560e8 55%,#2bbaa9); color:#fff; font-size:1.35rem; font-weight:800;
        box-shadow: 0 8px 20px rgba(72,97,230,.25); }
      .nova-brand-name { font:800 1.15rem 'Manrope',sans-serif; color:#fff; letter-spacing:-.03em; }
      .nova-brand-sub { color:#8f9db6; font-size:.71rem; letter-spacing:.13em; text-transform:uppercase; margin-top:.12rem; }
      .nova-sidebar-label { font-size:.68rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; color:#8291ac; margin:.5rem 0 .55rem; }
      .nova-hero { position:relative; overflow:hidden; background:linear-gradient(115deg,#192747 0%,#263c70 58%,#2f5c8f 100%);
        border-radius:22px; padding:1.55rem 1.7rem; color:white; margin-bottom:1.25rem; box-shadow:0 10px 28px rgba(24,43,83,.11); }
      .nova-hero:after { content:''; position:absolute; width:250px; height:250px; right:-65px; top:-130px; border:1px solid rgba(255,255,255,.14); border-radius:50%; box-shadow:0 0 0 32px rgba(255,255,255,.035),0 0 0 65px rgba(255,255,255,.025); }
      .nova-hero-eyebrow { font-size:.72rem; text-transform:uppercase; letter-spacing:.14em; color:#b8c8ef; font-weight:800; margin-bottom:.5rem; }
      .nova-hero-title { font:800 clamp(1.65rem,2.6vw,2.45rem) 'Manrope',sans-serif; letter-spacing:-.045em; line-height:1.12; color:#fff; margin-bottom:.62rem; max-width:800px; }
      .nova-hero-copy { color:#d6e0f4; font-size:.97rem; line-height:1.55; max-width:760px; }
      .nova-hero-tag { position:relative; z-index:1; display:inline-flex; align-items:center; gap:.42rem; padding:.42rem .68rem;
        border-radius:999px; color:#e8efff; background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.15); font-size:.72rem; font-weight:700; }
      .nova-section-kicker { color:#6c7da0; font-size:.7rem; text-transform:uppercase; letter-spacing:.13em; font-weight:800; margin-bottom:.28rem; }
      .nova-section-title { font:800 1.22rem 'Manrope',sans-serif; color:#1a2740; letter-spacing:-.03em; margin-bottom:.22rem; }
      .nova-section-copy { color:#728098; font-size:.88rem; line-height:1.5; margin-bottom:.85rem; }
      .nova-card { background:#fff; border:1px solid var(--nova-line); border-radius:16px; padding:1rem 1.05rem; height:100%; box-shadow:0 4px 15px rgba(26,39,72,.025); }
      .nova-card-title { font:750 1rem 'Manrope',sans-serif; color:#1b2943; margin-bottom:.38rem; }
      .nova-card-copy { color:#748199; font-size:.84rem; line-height:1.5; }
      .nova-icon-chip { display:inline-flex; width:35px; height:35px; align-items:center; justify-content:center; border-radius:11px;
        background:#eef2ff; color:#3e5dd5; font-size:1rem; margin-bottom:.65rem; }
      .nova-pill { display:inline-flex; padding:.27rem .58rem; border-radius:999px; font-size:.7rem; font-weight:800; letter-spacing:.015em; }
      .pill-live { background:#def7ed; color:#087f5b; }
      .pill-warn { background:#fff1d6; color:#96620a; }
      .pill-blue { background:#e6edff; color:#3754bd; }
      .pill-red { background:#ffe6e6; color:#b42318; }
      .nova-source { background:#fff; border:1px solid var(--nova-line); border-radius:13px; padding:.8rem .95rem; margin:.45rem 0; }
      .nova-source-title { color:#202e49; font-weight:750; margin-bottom:.3rem; }
      .nova-muted { color:#76839a; font-size:.8rem; }
      .nova-footer { border-top:1px solid #e6eaf2; margin-top:2rem; padding-top:1rem; color:#8490a4; font-size:.75rem; }
      .nova-empty { background:#fff; border:1px dashed #cbd5e1; border-radius:15px; padding:1.3rem; text-align:center; color:#718096; }
      @media (max-width: 750px) {
        .block-container { padding-top: .9rem; }
        .nova-hero { padding:1.2rem; border-radius:17px; }
        .nova-hero-title { font-size:1.7rem; }
        .nova-hero:after { opacity:.35; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# API utilities
# -----------------------------------------------------------------------------

def api_request(
    method: str,
    endpoint: str,
    *,
    timeout: int = 35,
    **kwargs: Any,
) -> tuple[Any | None, str | None]:
    """Make an API request; return data/error without interrupting the UI."""
    try:
        response = requests.request(
            method=method.upper(),
            url=f"{API_BASE_URL}{endpoint}",
            timeout=timeout,
            **kwargs,
        )
    except requests.Timeout:
        return None, "The API took too long to respond. A free-hosted service may be waking up; retry in a moment."
    except requests.RequestException as exc:
        return None, f"Could not connect to the API: {exc}"

    try:
        payload = response.json()
    except ValueError:
        payload = response.text

    if not response.ok:
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        message = str(detail)
        if response.status_code == 503:
            message = "The AI provider is temporarily unavailable (503). Try again shortly."
        elif response.status_code == 429:
            message = "The service has reached a rate limit. Wait briefly and retry."
        return None, f"API returned {response.status_code}: {message}"

    return payload, None


def show_api_error(message: str | None, heading: str = "Request not completed") -> None:
    if not message:
        return
    st.error(f"**{heading}**  \n{message}")
    with st.expander("Troubleshooting", expanded=False):
        st.write("Confirm the Render backend is awake and `API_BASE_URL` points to its root URL.")
        st.code(API_BASE_URL)
        st.link_button("Open API documentation", API_DOCS_URL)


def fetch_tickets() -> tuple[list[dict[str, Any]] | None, str | None]:
    payload, error = api_request("GET", "/tickets", timeout=25)
    if error:
        return None, error
    if not isinstance(payload, list):
        return None, "The ticket endpoint returned an unexpected response format."
    return payload, None


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def section_heading(kicker: str, title: str, copy: str = "") -> None:
    st.markdown(
        f'<div class="nova-section-kicker">{esc(kicker)}</div>'
        f'<div class="nova-section-title">{esc(title)}</div>'
        f'<div class="nova-section-copy">{esc(copy)}</div>' if copy else
        f'<div class="nova-section-kicker">{esc(kicker)}</div>'
        f'<div class="nova-section-title">{esc(title)}</div>',
        unsafe_allow_html=True,
    )


def hero(title: str, copy: str, tag: str = "SUPPORT OPERATIONS") -> None:
    st.markdown(
        f"""
        <div class="nova-hero">
          <div class="nova-hero-eyebrow">{esc(tag)}</div>
          <div class="nova-hero-title">{esc(title)}</div>
          <div class="nova-hero-copy">{esc(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def navigate_to(page: str) -> None:
    """Change sidebar navigation before the next Streamlit rerun."""
    st.session_state["nova_page_nav"] = page


def sentiment_pill(label: str) -> str:
    normalized = str(label or "UNKNOWN").upper()
    css = "pill-blue"
    if normalized == "NEGATIVE":
        css = "pill-red"
    elif normalized == "POSITIVE":
        css = "pill-live"
    return f'<span class="nova-pill {css}">{esc(normalized)}</span>'


def priority_pill(label: str) -> str:
    normalized = str(label or "UNKNOWN").upper()
    css = "pill-blue"
    if normalized == "HIGH":
        css = "pill-red"
    elif normalized == "MEDIUM":
        css = "pill-warn"
    elif normalized == "LOW":
        css = "pill-live"
    return f'<span class="nova-pill {css}">{esc(normalized)}</span>'


def display_prediction(result: dict[str, Any]) -> None:
    st.markdown("<div style='height:.25rem'></div>", unsafe_allow_html=True)
    section_heading("MODEL OUTPUT", "Ticket intelligence", "Predictions generated from the title and description.")

    category_col, priority_col, sentiment_col = st.columns(3)
    category_col.metric("Predicted category", result.get("predicted_category", "Unknown"))
    priority_col.metric("Priority", result.get("priority", "Unknown"))
    sentiment_col.metric("Customer sentiment", result.get("sentiment_label", "Unknown"))

    score = result.get("sentiment_score")
    if score is not None:
        score_value = max(0.0, min(1.0, float(score)))
        st.markdown("**Sentiment strength**")
        st.progress(score_value, text=f"Score: {score_value:.2f} · not a calibrated probability")

    ticket_id = result.get("id")
    if ticket_id is not None:
        st.success(f"Ticket #{ticket_id} saved successfully.")
    elif result.get("saved") is False:
        st.info("Analysis complete. This result has not been saved to the ticket queue.")


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("No source articles were returned for this query.")
        return

    for index, source in enumerate(sources, start=1):
        question = str(source.get("question", f"Source {index}"))
        category = str(source.get("category", "General"))
        score = source.get("similarity_score")
        with st.expander(f"{index:02d}  ·  {question}", expanded=(index == 1)):
            st.markdown(
                f'<span class="nova-pill pill-blue">{esc(category)}</span>',
                unsafe_allow_html=True,
            )
            st.write(source.get("answer", "No answer text available."))
            if score is not None:
                st.caption(f"TF-IDF similarity score: {float(score):.4f} · ranking signal, not a probability")


# -----------------------------------------------------------------------------
# Sidebar and navigation
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="nova-brand">
          <div class="nova-logo">N</div>
          <div>
            <div class="nova-brand-name">NOVA Support</div>
            <div class="nova-brand-sub">Intelligence workspace</div>
          </div>
        </div>
        <div class="nova-sidebar-label">WORKSPACE</div>
        """,
        unsafe_allow_html=True,
    )

    pages = {
        "Overview": "◫  Overview",
        "Ticket Analysis": "⊕  Ticket analysis",
        "AI Response Studio": "✳  AI response studio",
        "Ticket Queue": "▤  Ticket queue",
        "About Project": "◇  About project",
    }
    selected_page = st.radio(
        "Workspace navigation",
        options=list(pages.keys()),
        format_func=lambda page: pages[page],
        label_visibility="collapsed",
        key="nova_page_nav",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="nova-sidebar-label">SYSTEM</div>', unsafe_allow_html=True)
    st.caption("Backend endpoint")
    st.code(API_BASE_URL, language=None)

    if st.button("Check API connection", use_container_width=True, key="check_api_connection"):
        health, error = api_request("GET", "/health", timeout=12)
        if error:
            st.error("API unavailable")
            st.caption(error)
        elif isinstance(health, dict) and health.get("status") == "healthy":
            st.success("API is healthy")
        else:
            st.warning("API responded, but the health payload was unexpected.")

    st.link_button("Open API documentation ↗", API_DOCS_URL, use_container_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Portfolio demo · AI-generated text requires human review")


# -----------------------------------------------------------------------------
# Page: Overview
# -----------------------------------------------------------------------------

if selected_page == "Overview":
    health, health_error = api_request("GET", "/health", timeout=12)
    tickets, tickets_error = fetch_tickets()

    status_tag = "API CONNECTED" if not health_error and isinstance(health, dict) and health.get("status") == "healthy" else "API CHECK REQUIRED"
    hero(
        "Support operations, made clearer.",
        "A unified workspace for ticket triage, customer sentiment, knowledge retrieval, and AI-assisted response drafting.",
        status_tag,
    )

    if health_error:
        st.warning("The UI loaded, but the backend health check did not succeed. It may be waking up on the free tier.")
    if tickets_error:
        show_api_error(tickets_error, "Could not load ticket metrics")

    ticket_rows = tickets or []
    df = pd.DataFrame(ticket_rows)
    total_tickets = len(ticket_rows)
    high_priority = sum(str(row.get("priority", "")).upper() == "HIGH" for row in ticket_rows)
    negative_sentiment = sum(str(row.get("sentiment_label", "")).upper() == "NEGATIVE" for row in ticket_rows)
    known_categories = len({str(row.get("predicted_category", "Unknown")) for row in ticket_rows}) if ticket_rows else 0

    metric_cols = st.columns(4)
    metric_cols[0].metric("Tickets processed", f"{total_tickets:,}", help="Total tickets returned by the API.")
    metric_cols[1].metric("High priority", f"{high_priority:,}", help="Tickets currently predicted as High priority.")
    metric_cols[2].metric("Negative sentiment", f"{negative_sentiment:,}", help="Tickets with a Negative sentiment label.")
    metric_cols[3].metric("Categories seen", f"{known_categories:,}", help="Distinct predicted categories in the returned ticket list.")

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        section_heading("DISTRIBUTION", "Tickets by category", "A live view of predicted ticket categories.")
        if ticket_rows:
            category_series = pd.Series(
                [str(row.get("predicted_category", "Unknown")) for row in ticket_rows]
            ).value_counts().rename_axis("Category").to_frame("Tickets")
            st.bar_chart(category_series, height=220)
        else:
            st.markdown('<div class="nova-empty">Category analytics appear after tickets are saved.</div>', unsafe_allow_html=True)
    with chart_right:
        section_heading("TRIAGE", "Priority mix", "Use this view to spot the current volume of high-priority tickets.")
        if ticket_rows:
            priority_series = pd.Series(
                [str(row.get("priority", "Unknown")) for row in ticket_rows]
            ).value_counts().rename_axis("Priority").to_frame("Tickets")
            st.bar_chart(priority_series, height=220)
        else:
            st.markdown('<div class="nova-empty">Priority analytics appear after tickets are saved.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.25, 1], gap="large")

    with left:
        section_heading("WORKFLOW", "From issue to suggested action", "Use the tools below to move from raw customer text to a useful support workflow.")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown(
                """
                <div class="nova-card">
                  <div class="nova-icon-chip">⌕</div>
                  <div class="nova-card-title">01 · Triage a ticket</div>
                  <div class="nova-card-copy">Predict the ticket category, priority, and sentiment before deciding what to handle first.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                "Open ticket analysis →",
                use_container_width=True,
                key="go_analysis",
                on_click=navigate_to,
                args=("Ticket Analysis",),
            )
        with c2:
            st.markdown(
                """
                <div class="nova-card">
                  <div class="nova-icon-chip">✳</div>
                  <div class="nova-card-title">02 · Draft a response</div>
                  <div class="nova-card-copy">Retrieve knowledge-base articles and use Gemini to prepare a response for human review.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                "Open AI response studio →",
                use_container_width=True,
                key="go_ai",
                on_click=navigate_to,
                args=("AI Response Studio",),
            )

        section_heading("CAPABILITIES", "Built for a focused support workflow", "The current demo connects these components through the live FastAPI backend.")
        st.markdown(
            """
            <div class="nova-card">
              <div class="nova-card-title">Ticket intelligence</div>
              <div class="nova-card-copy">Scikit-learn classification and priority models · VADER sentiment analysis · SQLite ticket storage · TF-IDF retrieval · Gemini response generation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        section_heading("QUEUE SNAPSHOT", "Recent ticket activity", "Latest records returned by the backend.")
        if not ticket_rows:
            st.markdown('<div class="nova-empty">No tickets are available yet. Create one from Ticket analysis to populate this area.</div>', unsafe_allow_html=True)
        else:
            recent = ticket_rows[:6]
            for row in recent:
                ticket_id = row.get("id", "—")
                title = esc(row.get("title", "Untitled ticket"))
                category = esc(row.get("predicted_category", "Unknown"))
                priority = priority_pill(str(row.get("priority", "Unknown")))
                sentiment = sentiment_pill(str(row.get("sentiment_label", "Unknown")))
                created_at = esc(row.get("created_at", "Date unavailable"))
                st.markdown(
                    f"""
                    <div class="nova-source">
                      <div style="display:flex;justify-content:space-between;gap:.7rem;align-items:center;">
                        <div class="nova-source-title">#{esc(ticket_id)} · {title}</div>
                        {priority}
                      </div>
                      <div class="nova-muted" style="margin:.3rem 0 .5rem;">{category} &nbsp;·&nbsp; {created_at}</div>
                      {sentiment}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.button(
                "View full ticket queue →",
                use_container_width=True,
                key="go_queue",
                on_click=navigate_to,
                args=("Ticket Queue",),
            )

    st.markdown(
        '<div class="nova-footer">NOVA Support Intelligence · Portfolio demo · Predictions are assistive signals, not final decisions.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Page: Ticket analysis
# -----------------------------------------------------------------------------

elif selected_page == "Ticket Analysis":
    hero(
        "Turn customer text into ticket intelligence.",
        "Analyze a support issue, review model outputs, and optionally save the evaluated ticket to the queue.",
        "TICKET TRIAGE",
    )

    left, right = st.columns([1.05, .95], gap="large")
    with left:
        section_heading("INPUT", "Customer issue", "Use the customer's issue as written. Avoid including passwords or sensitive personal data.")
        with st.form("nova_ticket_form", clear_on_submit=False):
            title = st.text_input("Ticket title", placeholder="e.g. Office Wi-Fi disconnecting")
            description = st.text_area(
                "Issue description",
                placeholder="Describe what happened, who or what is affected, and any error message…",
                height=165,
            )
            button_col1, button_col2 = st.columns(2)
            analyze_clicked = button_col1.form_submit_button("Analyze ticket", type="primary", use_container_width=True)
            save_clicked = button_col2.form_submit_button("Analyze & save", use_container_width=True)

        if analyze_clicked or save_clicked:
            if not title.strip() or not description.strip():
                st.warning("Enter both a title and a description to continue.")
            else:
                with st.spinner("Running category, priority, and sentiment analysis…"):
                    if save_clicked:
                        result, error = api_request(
                            "POST",
                            "/tickets",
                            json={"title": title.strip(), "description": description.strip()},
                            timeout=40,
                        )
                    else:
                        result, error = api_request(
                            "POST",
                            "/tickets/predict",
                            json={"title": title.strip(), "description": description.strip()},
                            timeout=40,
                        )
                if error:
                    show_api_error(error, "Ticket analysis failed")
                elif isinstance(result, dict):
                    st.session_state["nova_last_prediction"] = result
                    if save_clicked:
                        st.session_state["nova_last_saved_ticket"] = result
                        st.success(f"Ticket #{result.get('id', '—')} saved to the queue.")

    with right:
        section_heading("RESULTS", "Prediction summary", "Results stay visible during this session so you can review them before moving on.")
        result = st.session_state.get("nova_last_prediction")
        if isinstance(result, dict):
            display_prediction(result)
            with st.expander("View submitted ticket text"):
                st.markdown("**Title**")
                st.write(result.get("title", "—"))
                st.markdown("**Description**")
                st.write(result.get("description", "—"))
        else:
            st.markdown(
                '<div class="nova-empty"><div style="font-size:1.5rem;margin-bottom:.4rem;">✦</div><b>Results will appear here</b><br><span class="nova-muted">Submit a ticket to see category, priority, and sentiment.</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    section_heading("EXAMPLE", "Try a sample ticket", "A quick starting point for a live demo.")
    sample_col1, sample_col2 = st.columns([1, 2])
    sample_col1.code("Title: Internet is not working", language=None)
    sample_col2.code("Description: I am frustrated because the office network keeps disconnecting.", language=None)


# -----------------------------------------------------------------------------
# Page: AI response studio
# -----------------------------------------------------------------------------

elif selected_page == "AI Response Studio":
    hero(
        "Draft a helpful response, grounded in your knowledge base.",
        "Retrieve relevant troubleshooting articles, send the context to Gemini, and review the generated draft before using it.",
        "AI RESPONSE STUDIO",
    )

    left, right = st.columns([.92, 1.08], gap="large")
    with left:
        section_heading("CUSTOMER CONTEXT", "Describe the customer's issue", "More specific issue descriptions usually make the retrieval step more useful.")
        with st.form("nova_ai_response_form"):
            query = st.text_area(
                "Customer message",
                placeholder="e.g. My office Wi-Fi disconnects every few minutes and I cannot finish my work…",
                height=180,
            )
            top_k = st.slider("Knowledge articles to retrieve", min_value=1, max_value=10, value=3)
            generate_clicked = st.form_submit_button("Generate response draft", type="primary", use_container_width=True)

        if generate_clicked:
            if not query.strip():
                st.warning("Enter the customer's issue first.")
            else:
                with st.spinner("Searching knowledge base and asking Gemini to draft a response…"):
                    result, error = api_request(
                        "GET",
                        "/knowledge/response",
                        params={"q": query.strip(), "top_k": top_k},
                        timeout=100,
                    )
                if error:
                    show_api_error(error, "Response generation failed")
                elif isinstance(result, dict):
                    st.session_state["nova_last_ai_response"] = result
                    st.session_state["nova_draft_edit"] = str(result.get("draft_response", ""))

    with right:
        section_heading("DRAFT PREVIEW", "Suggested customer response", "This is a draft. Review accuracy, tone, and policy before sending it to a customer.")
        ai_result = st.session_state.get("nova_last_ai_response")
        if isinstance(ai_result, dict):
            draft = str(ai_result.get("draft_response", ""))
            st.text_area("Editable response draft", value=draft, height=220, key="nova_draft_edit")
            st.caption("The response above is editable locally; changes are not sent or saved by this dashboard.")
            st.code(st.session_state.get("nova_draft_edit", draft), language=None)
            sources = ai_result.get("sources", [])
            st.markdown("**Retrieved sources**")
            if isinstance(sources, list):
                render_sources(sources)
        else:
            st.markdown(
                '<div class="nova-empty"><div style="font-size:1.5rem;margin-bottom:.4rem;">✳</div><b>Your response draft will appear here</b><br><span class="nova-muted">The AI will return a draft and the knowledge articles used as context.</span></div>',
                unsafe_allow_html=True,
            )

    st.info("This workflow uses TF-IDF to retrieve knowledge-base articles and Gemini to draft a response. Retrieval scores are ranking signals, not confidence probabilities.")


# -----------------------------------------------------------------------------
# Page: Ticket queue
# -----------------------------------------------------------------------------

elif selected_page == "Ticket Queue":
    hero(
        "A clear view of saved support tickets.",
        "Search, filter, inspect, and export tickets returned by the backend. Refresh to retrieve the latest records.",
        "TICKET QUEUE",
    )

    top_left, top_right = st.columns([1, 4])
    with top_left:
        refresh_clicked = st.button("Refresh queue", type="primary", use_container_width=True)
    if refresh_clicked or "nova_ticket_queue" not in st.session_state:
        with st.spinner("Loading tickets from the API…"):
            data, error = fetch_tickets()
        if error:
            show_api_error(error, "Could not load ticket queue")
        else:
            st.session_state["nova_ticket_queue"] = data or []

    ticket_rows = st.session_state.get("nova_ticket_queue", [])
    if not ticket_rows:
        st.markdown('<div class="nova-empty"><b>No saved tickets found</b><br><span class="nova-muted">Analyze and save a ticket to populate the queue.</span></div>', unsafe_allow_html=True)
    else:
        df = pd.DataFrame(ticket_rows)
        for column in ["id", "title", "description", "predicted_category", "priority", "sentiment_label", "sentiment_score", "created_at"]:
            if column not in df.columns:
                df[column] = ""

        total_col, high_col, neg_col = st.columns(3)
        total_col.metric("Total tickets", len(df))
        high_col.metric("High priority", int((df["priority"].astype(str).str.upper() == "HIGH").sum()))
        neg_col.metric("Negative sentiment", int((df["sentiment_label"].astype(str).str.upper() == "NEGATIVE").sum()))

        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        search_col, category_col, priority_col, sentiment_col = st.columns([1.6, 1, 1, 1])
        search_text = search_col.text_input("Search tickets", placeholder="ID, title, or description", key="nova_ticket_search")
        categories = sorted([str(x) for x in df["predicted_category"].dropna().unique() if str(x).strip()])
        priorities = sorted([str(x) for x in df["priority"].dropna().unique() if str(x).strip()])
        sentiments = sorted([str(x) for x in df["sentiment_label"].dropna().unique() if str(x).strip()])
        selected_category = category_col.selectbox("Category", ["All"] + categories)
        selected_priority = priority_col.selectbox("Priority", ["All"] + priorities)
        selected_sentiment = sentiment_col.selectbox("Sentiment", ["All"] + sentiments)

        filtered = df.copy()
        if search_text.strip():
            needle = search_text.strip().lower()
            mask = (
                filtered["id"].astype(str).str.lower().str.contains(needle, na=False)
                | filtered["title"].astype(str).str.lower().str.contains(needle, na=False)
                | filtered["description"].astype(str).str.lower().str.contains(needle, na=False)
            )
            filtered = filtered[mask]
        if selected_category != "All":
            filtered = filtered[filtered["predicted_category"].astype(str) == selected_category]
        if selected_priority != "All":
            filtered = filtered[filtered["priority"].astype(str) == selected_priority]
        if selected_sentiment != "All":
            filtered = filtered[filtered["sentiment_label"].astype(str) == selected_sentiment]

        st.caption(f"Showing {len(filtered)} of {len(df)} tickets")
        display_columns = ["id", "title", "predicted_category", "priority", "sentiment_label", "sentiment_score", "created_at"]
        table_df = filtered[display_columns].rename(columns={
            "id": "Ticket ID",
            "title": "Title",
            "predicted_category": "Category",
            "priority": "Priority",
            "sentiment_label": "Sentiment",
            "sentiment_score": "Sentiment strength",
            "created_at": "Created at",
        })
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        csv_bytes = table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export filtered tickets as CSV",
            data=csv_bytes,
            file_name=f"support_tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        section_heading("TICKET DETAIL", "Inspect a record", "Choose an ID to review its full description and model output.")
        ids = filtered["id"].tolist() if not filtered.empty else df["id"].tolist()
        if ids:
            chosen_id = st.selectbox("Ticket ID", ids, format_func=lambda value: f"Ticket #{value}")
            selected_rows = df[df["id"].astype(str) == str(chosen_id)]
            if not selected_rows.empty:
                row = selected_rows.iloc[0].to_dict()
                st.markdown(f"### {esc(row.get('title', 'Untitled ticket'))}")
                st.write(row.get("description", "No description available."))
                detail_col1, detail_col2, detail_col3 = st.columns(3)
                detail_col1.markdown(f"**Category**  \n{esc(row.get('predicted_category', 'Unknown'))}", unsafe_allow_html=True)
                detail_col2.markdown(f"**Priority**  \n{priority_pill(str(row.get('priority', 'Unknown')))}", unsafe_allow_html=True)
                detail_col3.markdown(f"**Sentiment**  \n{sentiment_pill(str(row.get('sentiment_label', 'Unknown')))}", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Page: About
# -----------------------------------------------------------------------------

elif selected_page == "About Project":
    hero(
        "An AI-assisted customer support workflow.",
        "A portfolio project demonstrating ticket ML, retrieval, LLM integration, API design, and a deployed web interface.",
        "PROJECT OVERVIEW",
    )

    left, right = st.columns([1.1, .9], gap="large")
    with left:
        section_heading("ARCHITECTURE", "How a request flows through the system", "The dashboard calls the public FastAPI backend; the backend owns prediction, storage, retrieval, and response generation.")
        steps = [
            ("01", "Streamlit dashboard", "Collects ticket text and displays model outputs."),
            ("02", "FastAPI service", "Validates requests and exposes prediction, ticket, and knowledge endpoints."),
            ("03", "Scikit-learn + VADER", "Predicts category and priority, then estimates sentiment polarity."),
            ("04", "TF-IDF retrieval", "Ranks knowledge-base articles against the customer's query."),
            ("05", "Gemini API", "Uses retrieved articles as context to draft a response."),
            ("06", "SQLite", "Stores and returns ticket records for the demo."),
        ]
        for number, title, copy in steps:
            st.markdown(
                f"""
                <div class="nova-source" style="display:flex;gap:.85rem;align-items:flex-start;">
                  <div class="nova-icon-chip" style="flex-shrink:0;margin:0;">{number}</div>
                  <div><div class="nova-source-title">{esc(title)}</div><div class="nova-card-copy">{esc(copy)}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right:
        section_heading("TECH STACK", "Tools used", "The app is deployed as a portfolio demo using free hosting tiers.")
        tech_items = [
            ("Python", "Core application language"),
            ("FastAPI", "REST API backend"),
            ("Streamlit", "Interactive web interface"),
            ("Scikit-learn", "Ticket classification, priority model, TF-IDF"),
            ("VADER", "Lightweight sentiment analysis"),
            ("Google Gemini API", "LLM response drafting"),
            ("SQLite", "Demo ticket storage"),
            ("Docker", "Local containerization"),
            ("Render + Streamlit Community Cloud", "Public demo deployment"),
        ]
        for name, description in tech_items:
            st.markdown(
                f'<div class="nova-source"><div class="nova-source-title">{esc(name)}</div><div class="nova-card-copy">{esc(description)}</div></div>',
                unsafe_allow_html=True,
            )

    st.info("Demo considerations: the ticket model is a prediction aid, the TF-IDF similarity score is not a confidence probability, generated responses need human review, and free-tier SQLite storage is not durable production storage.")
    link1, link2 = st.columns(2)
    with link1:
        st.link_button("View FastAPI docs ↗", API_DOCS_URL, use_container_width=True)
    with link2:
        st.link_button("View GitHub repository ↗", "https://github.com/NITESH864/customer-support-ai", use_container_width=True)


# Footer shown for each page
st.markdown(
    '<div class="nova-footer">NOVA Support Intelligence · Built with FastAPI, Streamlit, Scikit-learn, VADER, SQLite, and Gemini · Human review remains part of the workflow.</div>',
    unsafe_allow_html=True,
)
