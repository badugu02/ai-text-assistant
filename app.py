
import logging
import os
import time

import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ai_text_assistant")

MODEL = "gemini-3.5-flash"


# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
logger.info("AI Text Assistant started")

# Configure Gemini
api_key = None

# First, try Streamlit secrets
try:
    api_key = st.secrets.get("Gemini_API_Key")
except Exception:
    logger.info("Streamlit secrets file not found; checking environment variables")

# Fall back to .env
if not api_key:
    api_key = os.getenv("Gemini_API_Key")

if not api_key:
    logger.error("Gemini_API_Key is not configured")
    st.error(
        "Gemini API key not found. Add Gemini_API_Key to your .env "
        "file or .streamlit/secrets.toml."
    )
    st.stop()

logger.info("Gemini API client configured")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=30_000,
        retry_options=types.HttpRetryOptions(
            attempts=1,
            initial_delay=1,
            max_delay=4,
        ),
    ),
)


# Streamlit page
st.title("🤖 AI Text Assistant")
st.write("Ask me anything!")

# Get user input
question = st.text_input("Enter your question")

# Generate response
if st.button("Ask AI"):
    if question:
        logger.info("Generating AI response with model %s; question length=%d", MODEL, len(question))
        with st.spinner("AI is thinking..."):
            started_at = time.monotonic()
            try:
                logger.info("Sending streaming request to Gemini API")
                response_placeholder = st.empty()
                response_text = ""
                first_token_at = None

                for chunk in client.models.generate_content_stream(
                    model=MODEL,
                    contents=question,
                ):
                    chunk_text = chunk.text or ""
                    if chunk_text:
                        if first_token_at is None:
                            first_token_at = time.monotonic() - started_at
                            logger.info("First response token received in %.2f seconds", first_token_at)
                            response_placeholder.subheader("AI Response")
                        response_text += chunk_text
                        response_placeholder.markdown(response_text + "▌")

                elapsed = time.monotonic() - started_at
                response_placeholder.markdown(response_text)
                logger.info("Gemini streaming request completed in %.2f seconds", elapsed)
                logger.info("AI response generated successfully (%d characters)", len(response_text))

            except Exception as exc:
                elapsed = time.monotonic() - started_at
                logger.warning("Streaming request failed after %.2f seconds: %s", elapsed, exc)
                fallback_succeeded = False

                if "timeout" in str(exc).lower() or exc.__class__.__name__ == "ReadTimeout":
                    logger.info("Retrying with the synchronous Gemini API")
                    try:
                        fallback_response = client.models.generate_content(
                            model=MODEL,
                            contents=question,
                        )
                    except Exception:
                        logger.exception("Synchronous Gemini fallback also failed")
                    else:
                        fallback_succeeded = True
                        elapsed = time.monotonic() - started_at
                        logger.info("Synchronous fallback completed in %.2f seconds", elapsed)
                        st.subheader("AI Response")
                        st.markdown(fallback_response.text)

                if not fallback_succeeded:
                    logger.exception("Gemini request failed after %.2f seconds", elapsed)
                    st.error("Gemini did not respond in time or is currently unavailable. Please try again shortly.")
    else:
        logger.warning("Ask AI button clicked without a question")
        st.warning("Please enter a question.")
