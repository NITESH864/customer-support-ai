import os
import requests
import streamlit as st


# --------------------------------------------------
# 1. Dashboard configuration
# --------------------------------------------------

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
)

st.set_page_config(
    page_title="AI Customer Support",
    page_icon="🎧",
    layout="wide"
)


# --------------------------------------------------
# 2. Reusable API request function
# --------------------------------------------------

def call_api(method, endpoint, **kwargs):
    """Call the FastAPI backend and handle errors."""

    try:
        response = requests.request(
            method=method,
            url=f"{API_BASE_URL}{endpoint}",
            timeout=120,
            **kwargs
        )

        if not response.ok:
            try:
                detail = response.json().get(
                    "detail", response.text
                )
            except ValueError:
                detail = response.text

            st.error(
                f"API error ({response.status_code}): {detail}"
            )
            return None

        return response.json()

    except requests.RequestException as error:
        st.error(
            "FastAPI server se connection nahi ho pa raha. "
            "Pehle backend server start karo."
        )
        st.caption(str(error))
        return None


# --------------------------------------------------
# 3. Dashboard header and backend status
# --------------------------------------------------

st.title("🎧 AI Customer Support Intelligence")
st.caption(
    "Ticket classification · Priority prediction · "
    "Sentiment analysis · Knowledge search · AI response"
)

with st.sidebar:
    st.header("System Status")

    if st.button("Check Backend"):
        health = call_api("GET", "/health")

        if health:
            st.success("Backend is running")

    st.markdown("---")
    st.write("**Backend URL**")
    st.code(API_BASE_URL)

    st.write("**Technology Stack**")
    st.write("• FastAPI")
    st.write("• Scikit-learn")
    st.write("• Sentence Transformers")
    st.write("• Gemini API")
    st.write("• SQLite")
    st.write("• Streamlit")


# --------------------------------------------------
# 4. Create dashboard tabs
# --------------------------------------------------

ticket_tab, response_tab, saved_tab = st.tabs(
    [
        "🎫 Ticket Analysis",
        "🤖 AI Response",
        "📋 Saved Tickets"
    ]
)


# --------------------------------------------------
# 5. Ticket analysis and saving
# --------------------------------------------------

with ticket_tab:
    st.subheader("Analyze a Customer Ticket")

    st.write(
        "Enter the customer's issue to predict its category, "
        "priority, and sentiment."
    )

    with st.form("ticket_form"):
        title = st.text_input(
            "Ticket Title",
            placeholder="Example: Internet is not working"
        )

        description = st.text_area(
            "Ticket Description",
            placeholder=(
                "Describe the customer's problem here..."
            ),
            height=140
        )

        col1, col2 = st.columns(2)

        with col1:
            analyze_clicked = st.form_submit_button(
                "🔍 Analyze Ticket",
                use_container_width=True
            )

        with col2:
            save_clicked = st.form_submit_button(
                "💾 Save Ticket",
                use_container_width=True
            )

    # Analyze without saving
    if analyze_clicked:
        if not title.strip() or not description.strip():
            st.warning(
                "Please enter both ticket title and description."
            )
        else:
            with st.spinner("Analyzing ticket..."):
                result = call_api(
                    "POST",
                    "/tickets/predict",
                    json={
                        "title": title.strip(),
                        "description": description.strip()
                    }
                )

            if result:
                st.session_state["last_prediction"] = result

    # Analyze and save to database
    if save_clicked:
        if not title.strip() or not description.strip():
            st.warning(
                "Please enter both ticket title and description."
            )
        else:
            with st.spinner("Analyzing and saving ticket..."):
                result = call_api(
                    "POST",
                    "/tickets",
                    json={
                        "title": title.strip(),
                        "description": description.strip()
                    }
                )

            if result:
                st.session_state["last_saved_ticket"] = result
                st.session_state["last_prediction"] = result
                st.success(
                    f"Ticket saved successfully! ID: {result['id']}"
                )

    # Display latest prediction
    result = st.session_state.get("last_prediction")

    if result:
        st.markdown("---")
        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Category",
            result.get("predicted_category", "Unknown")
        )

        col2.metric(
            "Priority",
            result.get("priority", "Unknown")
        )

        col3.metric(
            "Sentiment",
            result.get("sentiment_label", "Unknown")
        )

        score = result.get("sentiment_score")

        if score is not None:
            st.write(
                f"Sentiment confidence: {float(score):.2%}"
            )

        if result.get("id") is not None:
            st.caption(f"Saved ticket ID: {result['id']}")
        else:
            st.info(
                "Prediction complete. This ticket has not "
                "been saved to the database."
            )


# --------------------------------------------------
# 6. AI response generation
# --------------------------------------------------

with response_tab:
    st.subheader("AI Customer Response Generator")

    st.write(
        "Search your knowledge base and generate a response "
        "draft for the customer."
    )

    with st.form("response_form"):
        query = st.text_area(
            "Customer Issue",
            placeholder=(
                "Example: My internet keeps disconnecting"
            ),
            height=120
        )

        top_k = st.slider(
            "Knowledge articles to retrieve",
            min_value=1,
            max_value=10,
            value=3
        )

        generate_clicked = st.form_submit_button(
            "✨ Generate Response",
            use_container_width=True
        )

    if generate_clicked:
        if not query.strip():
            st.warning("Please enter the customer's issue.")
        else:
            with st.spinner(
                "Searching knowledge base and generating draft..."
            ):
                result = call_api(
                    "GET",
                    "/knowledge/response",
                    params={
                        "q": query.strip(),
                        "top_k": top_k
                    }
                )

            if result:
                st.session_state["last_ai_response"] = result

    ai_result = st.session_state.get("last_ai_response")

    if ai_result:
        st.markdown("---")
        st.subheader("Suggested Customer Response")

        st.text_area(
            "Response Draft",
            value=ai_result.get("draft_response", ""),
            height=180,
            disabled=True,
            key="display_response_draft"
        )

        st.caption(
            "Review the draft before sending it to a customer."
        )

        if st.button("📋 Copy Draft"):
            st.code(ai_result.get("draft_response", ""))

        st.subheader("Retrieved Knowledge Sources")

        sources = ai_result.get("sources", [])

        if sources:
            for index, source in enumerate(sources, start=1):
                with st.expander(
                    f"{index}. {source.get('question', 'Article')}"
                ):
                    st.write("**Category:**", source.get("category"))
                    st.write("**Answer:**", source.get("answer"))
                    st.write(
                        "**Similarity:**",
                        source.get("similarity_score")
                    )
        else:
            st.info(
                "No sufficiently relevant knowledge articles "
                "were returned."
            )


# --------------------------------------------------
# 7. View saved tickets
# --------------------------------------------------

with saved_tab:
    st.subheader("Saved Support Tickets")

    st.write(
        "Retrieve tickets stored in the SQLite database."
    )

    if st.button(
        "🔄 Refresh Tickets",
        use_container_width=False
    ):
        with st.spinner("Loading tickets..."):
            tickets = call_api("GET", "/tickets")

        if tickets is not None:
            st.session_state["saved_tickets"] = tickets

    ticket_list = st.session_state.get("saved_tickets")

    if ticket_list is not None:
        if ticket_list:
            st.metric("Total Tickets", len(ticket_list))

            st.dataframe(
                ticket_list,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No tickets have been saved yet.")
    else:
        st.info(
            "Click 'Refresh Tickets' to load tickets "
            "from the database."
        )


# --------------------------------------------------
# 8. Footer
# --------------------------------------------------

st.markdown("---")
st.caption(
    "AI Customer Support Intelligence System | "
    "Review AI-generated responses before sending."
)
