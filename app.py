import streamlit as st
import requests

st.set_page_config(page_title="Vocal AI Analytics", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0B0F19; color: #E2E8F0; }
    
    .stButton>button {
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        color: white; border: none; border-radius: 8px; font-weight: 700; letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4); }
    
    .metric-card {
        background-color: #1E293B; border: 1px solid #334155; padding: 20px; border-radius: 12px; text-align: center;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 3.5rem; margin-top: 2rem; margin-bottom: 0;'>Acoustic Emotion Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 1.2rem;'>Understand the emotion behind the voice.</p>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.markdown("### 🎛️ Emotion Analyzer")
    uploaded_file = st.file_uploader("Upload an audio file (.wav format)", type=["wav"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        if st.button("Analyze Audio", use_container_width=True):
            with st.spinner("Listening to the audio..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
                    response = requests.post("http://127.0.0.1:8000/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        r_col1, r_col2 = st.columns(2)
                        with r_col1:
                            st.markdown(f"<div class='metric-card'><h4>Detected Emotion</h4><h2 style='color: #3B82F6;'>{result['predicted_emotion']}</h2></div>", unsafe_allow_html=True)
                        with r_col2:
                            st.markdown(f"<div class='metric-card'><h4>Processing Time</h4><h2 style='color: #8B5CF6;'>{result['latency_ms']} ms</h2></div>", unsafe_allow_html=True)
                            
                        st.markdown("<br>### Emotion Breakdown", unsafe_allow_html=True)
                        for emotion, percentage in result["confidence_scores"].items():
                            st.write(f"**{emotion.capitalize()}**")
                            st.progress(percentage / 100)
                    else:
                        st.error("Failed to analyze the audio.")
                except Exception as e:
                    st.error("Could not connect to the system.")

with col2:
    st.markdown("### 💡 Real-World Uses")
    st.info("**Customer Support:** Detect when a caller is getting frustrated so a human representative can step in and help immediately.")
    st.info("**Interview Practice:** Get automated feedback on your speaking tone. Find out if you sound confident or nervous before the real interview.")
    st.info("**Emergency Services:** Help operators quickly understand the stress level of a caller to prioritize urgent situations.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧠 How It Works")
    st.caption("This system listens to **how** you speak, not **what** you say. It analyzes the pitch, rhythm, and tone of the voice to determine the underlying emotional state, completely ignoring the actual dictionary words being spoken. This allows for unbiased and highly accurate emotion detection.")

st.divider()
st.markdown("<p style='text-align: center; color: #475569;'>Designed and Engineered by Tanishqa Mali</p>", unsafe_allow_html=True)