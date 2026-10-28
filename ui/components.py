"""
ui/components.py
Custom CSS and reusable layout elements.
"""

from typing import Optional
import streamlit as st


def apply_custom_css():
  st.markdown(
      """
        <style>
        .metric-card {
            background-color: #1e2129;
            border: 1px solid #313642;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 12px;
        }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            background-color: #2b313e;
            color: #6ee7b7;
            margin-right: 6px;
        }
        .stButton>button {
            border-radius: 6px;
        }
        </style>
    """,
      unsafe_allow_html=True,
  )


def render_starter_suggestions() -> Optional[str]:
  st.markdown("##### 💡 Suggested Research Inquiries")
  col1, col2 = st.columns(2)
  with col1:
    if st.button(
        "👤 Who is the author and what is the document title?",
        use_container_width=True,
    ):
      return (
          "Who is the author of this document and what is the official title?"
      )
    if st.button(
        "📖 What are the main chapters covered in this work?",
        use_container_width=True,
    ):
      return (
          "List all the main chapters and structural sections included in this"
          " document."
      )
  with col2:
    if st.button(
        "📐 Explain the core mathematical equations and formulas",
        use_container_width=True,
    ):
      return (
          "Explain the key mathematical models, formulas, and equations"
          " described in this research."
      )
    if st.button(
        "📊 Summarize the primary experimental results and conclusions",
        use_container_width=True,
    ):
      return (
          "Summarize the primary experimental results, findings, and final"
          " conclusions."
      )
  return None