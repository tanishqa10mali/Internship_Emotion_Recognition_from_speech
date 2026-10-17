
import os

import requests
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Vocal AI Analytics",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Local development fallback; Render uses BACKEND_URL.
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }

    .stButton > button {
        background: linear-gradient(
            90deg, #3B82F6 0%, #8B5CF6 100%
        );
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
    }

    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        overflow-wrap: anywhere;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <h1 style="
        text-align: center;
        font-size: 3.5rem;
        margin-top: 2rem;
        margin-bottom: 0;
    ">
        Acoustic Emotion Intelligence
    </h1>
    <p style="
        text-align: center;
        color: #94A3B8;
        font-size: 1.2rem;
    ">
        Understand the emotion behind the voice.
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()

col1, col2 = st.columns([1.5, 1], gap="large")


# --------------------------------------------------
# AUDIO ANALYZER
# --------------------------------------------------
with col1:
    st.markdown("### 🎛️ Emotion Analyzer")

    uploaded_file = st.file_uploader(
        "Upload an audio file (.wav format)",
        type=["wav"],
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("Analyze Audio", use_container_width=True):
            with st.spinner("Analyzing your audio..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "audio/wav",
                        )
                    }

                    response = requests.post(
                        f"{BACKEND_URL}/predict",
                        files=files,
                        timeout=120,
                    )

                    if response.status_code == 200:
                        result = response.json()

                        predicted_emotion = result.get(
                            "predicted_emotion", "Unknown"
                        )
                        latency = result.get("latency_ms", "N/A")
                        confidence_scores = result.get(
                            "confidence_scores", {}
                        )

                        st.markdown("<br>", unsafe_allow_html=True)

                        r_col1, r_col2 = st.columns(2)

                        with r_col1:
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <h4>Detected Emotion</h4>
                                    <h2 style="color: #3B82F6;">
                                        {predicted_emotion}
                                    </h2>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with r_col2:
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <h4>Processing Time</h4>
                                    <h2 style="color: #8B5CF6;">
                                        {latency} ms
                                    </h2>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.markdown("### Emotion Breakdown")

                        if confidence_scores:
                            for emotion, percentage in confidence_scores.items():
                                try:
                                    score = float(percentage)
                                    st.write(
                                        f"**{str(emotion).capitalize()}** "
                                        f"— {score:.2f}%"
                                    )
                                    st.progress(
                                        max(0.0, min(score / 100, 1.0))
                                    )
                                except (ValueError, TypeError):
                                    st.write(f"**{emotion}**: {percentage}")
                        else:
                            st.info("No confidence scores were returned.")

                    else:
                        st.error(
                            f"Audio analysis failed "
                            f"(HTTP {response.status_code})."
                        )
                        with st.expander("View server response"):
                            st.code(response.text[:2000])

                except requests.exceptions.Timeout:
                    st.error(
                        "The analysis took too long. Please try again "
                        "with a shorter audio file."
                    )

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Could not connect to the backend. Please check "
                        "that the backend is running and BACKEND_URL "
                        "is configured correctly."
                    )

                except requests.exceptions.RequestException as exc:
                    st.error(f"Request failed: {exc}")

                except (ValueError, KeyError) as exc:
                    st.error(f"Invalid response from the backend: {exc}")


# --------------------------------------------------
# INFORMATION PANEL
# --------------------------------------------------
with col2:
    st.markdown("### 💡 Real-World Uses")

    st.info(
        "**Customer Support:** Detect when a caller may be getting "
        "frustrated so a human representative can step in and help."
    )

    st.info(
        "**Interview Practice:** Explore your speaking tone and "
        "reflect on how you sound before an interview."
    )

    st.info(
        "**Emergency Services:** Explore whether vocal stress "
        "signals may help operators assess calls. This should not "
        "replace trained human assessment."
    )

    st.markdown("### 🧠 How It Works")

    st.caption(
        "This system analyzes vocal characteristics such as pitch, "
        "rhythm, and tone to predict an emotion from audio. It is "
        "designed to analyze how a person speaks rather than the "
        "dictionary meaning of their words. Predictions are estimates "
        "and may not reflect a person's actual emotional state."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.markdown(
    """
    <p style="
        text-align: center;
        color: #475569;
    ">
        Designed and Engineered by Team Inteli6
    </p>
    """,
    unsafe_allow_html=True,
)
