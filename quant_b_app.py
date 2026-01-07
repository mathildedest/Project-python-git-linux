import streamlit as st

from quant_b_dashboard import run_quant_b_page


def main():
    """
    Entry point for the Quant B module (multi-asset portfolio).
    I keep it in a separate file so I don't modify app.py from Quant A.
    """
    st.title("Quant B - Multi-Asset Portfolio Dashboard")
    run_quant_b_page()


if __name__ == "__main__":
    main()
