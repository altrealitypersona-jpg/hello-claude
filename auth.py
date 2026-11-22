import hmac

import streamlit as st


def require_passcode():
    """Stop the page until the right passcode is entered.

    Once it's correct, st.session_state remembers it, so every page in this
    browser session opens without asking again.
    """
    if st.session_state.get("unlocked"):
        return

    # The passcode lives in .streamlit/secrets.toml, which git ignores.
    expected = st.secrets.get("APP_PASSCODE")
    if not expected:
        st.error("APP_PASSCODE is missing from .streamlit/secrets.toml.")
        st.stop()

    # type="password" hides what's typed.
    entered = st.text_input("Passcode", type="password")
    if entered:
        # compare_digest takes the same time whether the guess is close or not,
        # so timing can't hint at the right passcode.
        if hmac.compare_digest(entered.encode(), str(expected).encode()):
            st.session_state["unlocked"] = True
            # Run the page again from the top, now unlocked, so the passcode box goes away.
            st.rerun()
        st.error("Wrong passcode.")

    # Nothing below the require_passcode() call runs until the passcode is right.
    st.stop()
