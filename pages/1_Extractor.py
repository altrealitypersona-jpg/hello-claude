from typing import Literal

import anthropic
import streamlit as st
from pydantic import BaseModel, ConfigDict, ValidationError

# Same model as app.py. (We copy the line instead of importing app.py,
# because importing it would also run app.py's page code.)
MODEL = "claude-sonnet-5-5"

# The rule for missing values, used in the Nullable tool's descriptions.
NULL_RULE = "Use null if the document does not state this value. Never estimate, infer or calculate a missing value."

# Same instruction for every extraction request, including retries.
SYSTEM_PROMPT = "Always call the extract_document tool once with the data from the user's document."

# How many extra requests we make when the numbers don't add up.
MAX_RETRIES = 2

# Two amounts count as equal if they differ by at most this much (rounding on receipts).
TOLERANCE = 0.01

# Sent to Claude, together with the list of errors, when the numbers don't add up.
RETRY_INSTRUCTION = (
    "Re-read the document carefully. If the document itself shows these numbers, return them unchanged. "
    "Never change a value to make the arithmetic work."
)


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


def get_nullable_extraction_tool():
    """Same tool, but missing values come back as null instead of being guessed.

    "required" still lists every field: Claude must always send each key,
    but the value is allowed to be null.
    """
    line_item = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "What was bought or billed."},
            "quantity": {"type": ["number", "null"], "description": f"How many units. {NULL_RULE}"},
            "unit_price": {"type": ["number", "null"], "description": f"Price of one unit. {NULL_RULE}"},
            "amount": {"type": ["number", "null"], "description": f"Line total as printed. {NULL_RULE}"},
        },
        "required": ["description", "quantity", "unit_price", "amount"],
        "additionalProperties": False,
    }

    return {
        "name": "extract_document",
        "description": (
            "Record the structured data from a single receipt or invoice that the user pasted as text. "
            "Call this exactly once with the vendor, date, currency, every line item, subtotal, tax and total "
            f"as they appear in the document. {NULL_RULE}"
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "enum": ["receipt", "invoice", "other"],
                    "description": "Whether this is a receipt, an invoice, or something else.",
                },
                "document_type_detail": {
                    "type": ["string", "null"],
                    "description": (
                        "When document_type is 'other', a short description of what the document is "
                        f"(e.g. 'bank statement'). Use null when document_type is 'receipt' or 'invoice'. {NULL_RULE}"
                    ),
                },
                "vendor_name": {
                    "type": ["string", "null"],
                    "description": f"The business that issued the document. {NULL_RULE}",
                },
                "document_date": {
                    "type": ["string", "null"],
                    "description": f"Date on the document, as YYYY-MM-DD. {NULL_RULE}",
                },
                "currency": {
                    "type": ["string", "null"],
                    "description": f"3-letter currency code, e.g. USD, EUR, INR. {NULL_RULE}",
                },
                "line_items": {"type": "array", "items": line_item},
                "subtotal": {"type": ["number", "null"], "description": f"Total before tax. {NULL_RULE}"},
                "tax": {"type": ["number", "null"], "description": f"Tax amount. {NULL_RULE}"},
                "total": {"type": ["number", "null"], "description": f"Final amount due or paid. {NULL_RULE}"},
            },
            "required": [
                "document_type", "document_type_detail", "vendor_name", "document_date", "currency",
                "line_items", "subtotal", "tax", "total",
            ],
            "additionalProperties": False,
        },
    }


# Pydantic models: a second, local check that the data has the shape we expect.
# "float | None" with no default means: the key must be present, but may be null.
# extra="forbid" rejects any field we didn't define.
class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    quantity: float | None
    unit_price: float | None
    amount: float | None


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal["receipt", "invoice", "other"]
    # Has a default because Required mode never sends this field.
    document_type_detail: str | None = None
    vendor_name: str | None
    document_date: str | None
    currency: str | None
    line_items: list[LineItem]
    subtotal: float | None
    tax: float | None
    total: float | None


def find_missing_fields(extraction):
    """Names of every field that came back null, e.g. 'tax' or 'line_items[2].unit_price'."""
    missing = [
        name
        for name in ["vendor_name", "document_date", "currency", "subtotal", "tax", "total"]
        if getattr(extraction, name) is None
    ]
    # document_type_detail is only expected for "other" documents.
    if extraction.document_type == "other" and extraction.document_type_detail is None:
        missing.append("document_type_detail")
    for i, item in enumerate(extraction.line_items):
        for name in ["quantity", "unit_price", "amount"]:
            if getattr(item, name) is None:
                missing.append(f"line_items[{i}].{name}")
    return missing


def close_enough(a, b):
    return abs(a - b) <= TOLERANCE


def check_totals(extraction):
    """Check that the extracted numbers add up.

    Returns two lists:
    - errors: the numbers are stated but don't add up (worth asking Claude to re-read).
    - cannot_verify: a check couldn't run because a value is missing (not worth a retry).
    """
    errors = []
    cannot_verify = []

    subtotal, tax, total = extraction.subtotal, extraction.tax, extraction.total
    amounts = [item.amount for item in extraction.line_items]

    # Why the line items can't be added up, or None if they can.
    # (With no line items at all, the sum would be 0, which would wrongly fail the checks.)
    if not amounts:
        amounts_problem = "no line items found"
    elif any(amount is None for amount in amounts):
        amounts_problem = "a line item amount not stated"
    else:
        amounts_problem = None
    amounts_sum = None if amounts_problem else sum(amounts)

    # Rule 1: the line items add up to the subtotal.
    reasons = []
    if subtotal is None:
        reasons.append("subtotal not stated")
    if amounts_problem:
        reasons.append(amounts_problem)
    if reasons:
        cannot_verify.append(f"Cannot verify: {', '.join(reasons)} (line items vs subtotal)")
    elif not close_enough(amounts_sum, subtotal):
        errors.append(f"Line items add up to {amounts_sum:,.2f}, but the subtotal is {subtotal:,.2f}.")

    # Rule 2: subtotal + tax = total.
    reasons = [f"{name} not stated" for name, value in
               [("subtotal", subtotal), ("tax", tax), ("total", total)] if value is None]
    if reasons:
        cannot_verify.append(f"Cannot verify: {', '.join(reasons)} (subtotal + tax vs total)")
    elif not close_enough(subtotal + tax, total):
        errors.append(
            f"Subtotal {subtotal:,.2f} + tax {tax:,.2f} = {subtotal + tax:,.2f}, but the total is {total:,.2f}."
        )

    # Rule 3: with no subtotal and no tax, the line items add up to the total.
    if subtotal is None and tax is None:
        reasons = []
        if total is None:
            reasons.append("total not stated")
        if amounts_problem:
            reasons.append(amounts_problem)
        if reasons:
            cannot_verify.append(f"Cannot verify: {', '.join(reasons)} (line items vs total)")
        elif not close_enough(amounts_sum, total):
            errors.append(f"Line items add up to {amounts_sum:,.2f}, but the total is {total:,.2f}.")

    return errors, cannot_verify


def key_numbers(extraction):
    """The numbers the checks use. If two attempts give the same ones, retrying again won't help."""
    return (
        extraction.subtotal,
        extraction.tax,
        extraction.total,
        [item.amount for item in extraction.line_items],
    )


def find_tool_block(response):
    """The extract_document tool call in Claude's reply, or None if there isn't one."""
    return next(
        (b for b in response.content if b.type == "tool_use" and b.name == "extract_document"),
        None,
    )


def call_claude(messages, tool):
    """Send the conversation so far. On an API error, show a message and stop the page."""
    try:
        return client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            # This model doesn't allow forcing a tool, so we ask for it in plain words instead.
            system=SYSTEM_PROMPT,
            tools=[tool],
            # "auto" lets Claude choose; disable_parallel_tool_use caps it at one tool call.
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=messages,
            # If Claude declines for safety reasons, let the API retry on a suitable fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError:
        st.error("The API key was rejected. Check .streamlit/secrets.toml.")
    except anthropic.RateLimitError:
        st.error("Rate limited. Wait a moment and try again.")
    except anthropic.APIStatusError as e:
        st.error(f"API error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        st.error("Couldn't reach the API. Check your internet connection.")
    st.stop()


def show_text(value):
    return "—" if value is None else value


def show_money(value):
    return "—" if value is None else f"{value:,.2f}"


st.title("Receipt / Invoice Extractor")

schema_mode = st.sidebar.radio("Schema mode", ["Required", "Nullable"])
# For testing the retry loop: pretend the first line item was read 10x too big.
simulate_misread = st.sidebar.checkbox(
    "Simulate a misread",
    help="Multiplies the first line item's amount by 10 after the first extraction only.",
)

# The key comes from .streamlit/secrets.toml, which git ignores.
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

document_text = st.text_area("Paste a receipt or invoice", height=300)

if st.button("Extract"):
    if not document_text.strip():
        st.warning("Paste a receipt or invoice first.")
        st.stop()

    tool = get_extraction_tool() if schema_mode == "Required" else get_nullable_extraction_tool()

    # The conversation so far. Retries add to it; nothing earlier is ever changed.
    messages = [{"role": "user", "content": document_text}]

    with st.spinner("Extracting..."):
        response = call_claude(messages, tool)

    # "tool_use" is the normal value here: it means Claude called the tool.
    st.write(f"**stop_reason:** {response.stop_reason}")

    # Find the tool call in the reply. Its .input is the extracted data, already a dict.
    tool_block = find_tool_block(response)
    if tool_block is None:
        st.error("Claude didn't call extract_document, so there's nothing to show.")
        text = "".join(b.text for b in response.content if b.type == "text")
        if text:
            st.markdown(text)
        st.stop()

    # Check the data against our Pydantic model before using it.
    try:
        extraction = Extraction.model_validate(tool_block.input)
    except ValidationError as e:
        st.error(f"The extracted data didn't have the expected shape:\n\n```\n{e}\n```")
        with st.expander("Raw tool input"):
            st.json(tool_block.input)
        st.stop()

    # Simulated misread: change OUR copy of the first amount. What Claude actually sent stays
    # untouched in the conversation, so on a retry Claude re-reads the document and should
    # return the real number.
    misread_applied = False
    if simulate_misread and extraction.line_items and extraction.line_items[0].amount is not None:
        extraction = extraction.model_copy(deep=True)
        extraction.line_items[0].amount *= 10
        misread_applied = True

    errors, cannot_verify = check_totals(extraction)
    # One entry per attempt, shown in the "Attempt log" expander at the end.
    attempt_log = [{"attempt": 1, "errors": errors, "notes": cannot_verify, "misread": misread_applied}]
    retries_used = 0
    stop_message = None  # Why the retry loop ended early, if it did.

    # Retry only when numbers are stated but don't add up. Missing values (cannot_verify) never retry.
    while errors and retries_used < MAX_RETRIES:
        retries_used += 1

        # Add Claude's last reply exactly as it came back, then tell Claude its tool call
        # produced numbers that don't add up (is_error marks it as a failed result).
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "is_error": True,
                "content": "The extracted numbers don't add up:\n"
                           + "\n".join(f"- {error}" for error in errors)
                           + "\n\n" + RETRY_INSTRUCTION,
            }],
        })

        with st.spinner(f"Numbers don't add up. Asking Claude to re-read (retry {retries_used} of {MAX_RETRIES})..."):
            response = call_claude(messages, tool)

        new_tool_block = find_tool_block(response)
        if new_tool_block is None:
            stop_message = f"Retry {retries_used}: Claude didn't call extract_document, so we kept the previous result."
            attempt_log.append({"attempt": retries_used + 1, "errors": [stop_message], "notes": [], "misread": False})
            break

        # Same Pydantic check as the first attempt.
        try:
            new_extraction = Extraction.model_validate(new_tool_block.input)
        except ValidationError as e:
            stop_message = f"Retry {retries_used}: the data didn't have the expected shape, so we kept the previous result."
            attempt_log.append({"attempt": retries_used + 1, "errors": [stop_message, str(e)], "notes": [], "misread": False})
            break

        new_errors, new_notes = check_totals(new_extraction)
        attempt_log.append({"attempt": retries_used + 1, "errors": new_errors, "notes": new_notes, "misread": False})

        same_numbers = key_numbers(new_extraction) == key_numbers(extraction)
        tool_block, extraction, errors, cannot_verify = new_tool_block, new_extraction, new_errors, new_notes

        # Claude read the same numbers again: the document itself probably doesn't add up.
        if same_numbers:
            stop_message = "Claude returned the same numbers again, so we stopped retrying."
            break

    st.write(f"**Mode:** {schema_mode}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Type", extraction.document_type)
    col2.metric("Vendor", show_text(extraction.vendor_name))
    col3.metric("Date", show_text(extraction.document_date))
    col4.metric("Currency", show_text(extraction.currency))
    if extraction.document_type_detail:
        st.caption(f"Document type detail: {extraction.document_type_detail}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Subtotal", show_money(extraction.subtotal))
    col2.metric("Tax", show_money(extraction.tax))
    col3.metric("Total", show_money(extraction.total))

    st.subheader("Not stated in document")
    missing = find_missing_fields(extraction)
    if missing:
        st.markdown("\n".join(f"- `{name}`" for name in missing))
    else:
        st.write("Nothing missing.")

    st.subheader("Line items")
    st.dataframe([item.model_dump() for item in extraction.line_items])

    with st.expander("Extracted JSON"):
        st.json(extraction.model_dump())

    st.subheader("Arithmetic check")
    st.write(f"**Retries used:** {retries_used} of {MAX_RETRIES}")
    if stop_message:
        st.caption(stop_message)

    # Errors win over notes: numbers that don't add up need a person to look.
    if errors:
        st.badge("Needs human review: document arithmetic inconsistent", color="red")
        st.markdown("\n".join(f"- {error}" for error in errors))
    elif cannot_verify:
        st.badge("Cannot verify", color="orange")
        st.markdown("\n".join(f"- {note}" for note in cannot_verify))
    else:
        st.badge("Verified", color="green")

    with st.expander("Attempt log"):
        for entry in attempt_log:
            label = f"**Attempt {entry['attempt']}**"
            if entry["misread"]:
                label += " (simulated misread: first line item amount x10)"
            st.markdown(label)
            if entry["errors"]:
                st.markdown("\n".join(f"- {error}" for error in entry["errors"]))
            else:
                st.write("No errors.")
            if entry["notes"]:
                st.markdown("\n".join(f"- {note}" for note in entry["notes"]))
