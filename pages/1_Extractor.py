import anthropic
import streamlit as st

# Same model as app.py. (We copy the line instead of importing app.py,
# because importing it would also run app.py's page code.)
MODEL = "claude-sonnet-5-5"


def get_extraction_tool():
    """The one tool Claude can use on this page. Edit fields here."""
    line_item = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "What was bought or billed."},
            "quantity": {"type": "number", "description": "How many units."},
            "unit_price": {"type": "number", "description": "Price of one unit."},
            "amount": {"type": "number", "description": "Line total (usually quantity x unit_price)."},
        },
        "required": ["description", "quantity", "unit_price", "amount"],
        "additionalProperties": False,
    }

    return {
        "name": "extract_document",
        "description": (
            "Record the structured data from a single receipt or invoice that the user pasted as text. "
            "Call this exactly once with the vendor, date, currency, every line item, subtotal, tax and total "
            "as they appear in the document. Use only values from the document; do not invent or estimate them."
        ),
        # strict=True makes the API guarantee the input matches the schema below.
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "enum": ["receipt", "invoice", "other"],
                    "description": "Whether this is a receipt, an invoice, or something else.",
                },
                "vendor_name": {"type": "string", "description": "The business that issued the document."},
                "document_date": {"type": "string", "description": "Date on the document, as YYYY-MM-DD."},
                "currency": {"type": "string", "description": "3-letter currency code, e.g. USD, EUR, INR."},
                "line_items": {"type": "array", "items": line_item},
                "subtotal": {"type": "number", "description": "Total before tax."},
                "tax": {"type": "number", "description": "Tax amount."},
                "total": {"type": "number", "description": "Final amount due or paid."},
            },
            "required": [
                "document_type", "vendor_name", "document_date", "currency",
                "line_items", "subtotal", "tax", "total",
            ],
            "additionalProperties": False,
        },
    }


st.title("Receipt / Invoice Extractor")

# The key comes from .streamlit/secrets.toml, which git ignores.
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

document_text = st.text_area("Paste a receipt or invoice", height=300)

if st.button("Extract"):
    if not document_text.strip():
        st.warning("Paste a receipt or invoice first.")
        st.stop()

    try:
        with st.spinner("Extracting..."):
            response = client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                # This model doesn't allow forcing a tool, so we ask for it in plain words instead.
                system="Always call the extract_document tool once with the data from the user's document.",
                tools=[get_extraction_tool()],
                # "auto" lets Claude choose; disable_parallel_tool_use caps it at one tool call.
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                messages=[{"role": "user", "content": document_text}],
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

    # "tool_use" is the normal value here: it means Claude called the tool.
    st.write(f"**stop_reason:** {response.stop_reason}")

    # Find the tool call in the reply. Its .input is the extracted data, already a dict.
    tool_block = next(
        (b for b in response.content if b.type == "tool_use" and b.name == "extract_document"),
        None,
    )
    if tool_block is None:
        st.error("Claude didn't call extract_document, so there's nothing to show.")
        text = "".join(b.text for b in response.content if b.type == "text")
        if text:
            st.markdown(text)
        st.stop()

    data = tool_block.input

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Type", data["document_type"])
    col2.metric("Vendor", data["vendor_name"])
    col3.metric("Date", data["document_date"])
    col4.metric("Currency", data["currency"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Subtotal", f"{data['subtotal']:,.2f}")
    col2.metric("Tax", f"{data['tax']:,.2f}")
    col3.metric("Total", f"{data['total']:,.2f}")

    st.subheader("Line items")
    st.dataframe(data["line_items"])

    with st.expander("Extracted JSON"):
        st.json(data)
