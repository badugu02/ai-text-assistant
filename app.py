
import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_Key")

st.title("🤖 AI Text Assistant")
st.write("Summarize text, fix grammar, or explain concepts simply.")

if not api_key:
    st.error("Gemini API key not found. Check your .env file or Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

user_text = st.text_area("Enter your text here:")

task = st.selectbox(
    "Choose a task:",
    ["Summarize", "Fix Grammar", "Explain Simply"]
)

instructions = {
    "Summarize": "Summarize the following text clearly and briefly.",
    "Fix Grammar": "Correct the grammar and spelling while preserving the original meaning.",
    "Explain Simply": "Explain the following text in simple language with an example if useful."
}

if st.button("Generate"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Gemini is thinking..."):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"{instructions[task]}\n\nText:\n{user_text}"
                )

                st.subheader("AI Response")
                st.write(response.text)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
