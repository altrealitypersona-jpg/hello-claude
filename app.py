import anthropic
import streamlit as st

MODEL = "claude-sonnet-5-5"

st.title("Hello, Claude")

# The key comes from .streamlit/secrets.toml, which git ignores.
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

prompt = st.text_area("Your message")

if st.button("Send"):
    if not prompt.strip():
        st.warning("Type a message first.")
        st.stop()

    try:
        with st.spinner("Waiting for Claude..."):
            response = client.beta.messages.create(
                model=MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                # If Claude declines for safety reasons, let the API retry on a suitable fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
    except anthropic.AuthenticationError:
        st.error("The API key was rejected. Check .streamlit/secrets.toml.")
        st.stop()
    except anthropic.RateLimitError:
        st.error("Rate limited. Wait a moment and try again.")
        st.stop()
    except anthropic.APIStatusError as e:
        st.error(f"API error ({e.status_code}): {e.message}")
        st.stop()
    except anthropic.APIConnectionError:
        st.error("Couldn't reach the API. Check your internet connection.")
        st.stop()

    # The reply is a list of blocks; keep only the text ones.
    reply = "".join(block.text for block in response.content if block.type == "text")
    st.markdown(reply or "_(no text in the reply)_")

    st.write(f"**stop_reason:** {response.stop_reason}")
    st.write(f"**Input tokens:** {response.usage.input_tokens}")
    st.write(f"**Output tokens:** {response.usage.output_tokens}")

    with st.expander("Raw response"):
        st.json(response.to_dict())
