
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

st.title("🤖 AI Text Assistant")
st.write("Summarize text, fix grammar, or explain concepts simply.")

if not api_key:
    st.error("API key not found. Please check your .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

user_text = st.text_area("Enter your text here:")

task = st.selectbox(
    "Choose a task:",
    ["Summarize", "Fix Grammar", "Explain Simply"]
)

if st.button("Generate"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        instructions = {
            "Summarize": "Summarize the following text clearly and briefly.",
            "Fix Grammar": "Correct the grammar and spelling while preserving the original meaning.",
            "Explain Simply": "Explain the following text in simple language with an example if useful."
        }

        with st.spinner("AI is thinking..."):
            try:
                response = client.responses.create(
                    model="gpt-4.1-mini",
                    instructions=instructions[task],
                    input=user_text
                )

                st.subheader("AI Response")
                st.write(response.output_text)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
