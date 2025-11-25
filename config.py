"""Configuration and session state initialization"""
import uuid
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def init_session_state():
    """Initialize Streamlit session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "dual_chat_mode" not in st.session_state:
        st.session_state.dual_chat_mode = False
    if "chat_tabs" not in st.session_state:
        st.session_state.chat_tabs = ["Chat 1"]
    if "tab_counter" not in st.session_state:
        st.session_state.tab_counter = 1
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "active_tool_context" not in st.session_state:
        st.session_state.active_tool_context = None
    # File metadata will be loaded from persistent storage when needed
    # Don't initialize here - let document_handling load it

