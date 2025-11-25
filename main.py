"""Main Streamlit application entry point"""
import time
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from config import init_session_state
from llm_setup import add_document_retrieval_tool
from document_handling import build_context_blob, handle_pdf_upload, document_retrieval
from chat_handlers import ensure_conversation, render_dual_chat_mode

# Add document_retrieval tool to LLM tools list
add_document_retrieval_tool(document_retrieval)


if __name__ == "__main__":
    # Initialize session state
    init_session_state()
    
    # Page configuration
    st.set_page_config(page_title="Tool Augmented Chatbot", layout="wide")
    st.title("Tool Augmented Chatbot")
    
    # Dual-chat mode toggle
    st.sidebar.markdown("Enable Dual Chat Mode  \n(FAISS vs Direct)")
    dual_chat_mode = st.sidebar.toggle(
        "Enable Dual Chat Mode",
        value=st.session_state.dual_chat_mode,
        key="dual_chat_toggle",
        label_visibility="collapsed",
    )
    st.session_state.dual_chat_mode = dual_chat_mode
    
    if dual_chat_mode:
        st.warning(
            "🔬 Dual Chat Mode: Compare FAISS vector search vs direct document upload side-by-side",
            icon="⚠️",
        )
        render_dual_chat_mode()
    else:
        st.warning(
            "Ask me anything or 📱 On mobile? Tap >> to upload PDFs if it's document-related!",
            icon="⚠️",
        )

        # Use CSS to position button inline with tabs
        st.markdown("""
            <style>
            div[data-testid="stTabs"] {
                position: relative;
            }
            .inline-tab-button {
                position: absolute;
                right: 0;
                top: 0.5rem;
                z-index: 10;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Create tabs
        tabs = st.tabs(st.session_state.chat_tabs)
        
        # Place button in a container that will be positioned next to tabs
        with st.container():
            col1, col2 = st.columns([0.97, 0.03])
            with col2:
                st.markdown('<div class="inline-tab-button">', unsafe_allow_html=True)
                if st.button("➕", key="new_tab_button", help="Add new chat tab", use_container_width=True):
                    st.session_state.tab_counter += 1
                    new_tab_name = f"Chat {st.session_state.tab_counter}"
                    st.session_state.chat_tabs.append(new_tab_name)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        
        for tab_name, tab_container in zip(st.session_state.chat_tabs, tabs):
            conversation = ensure_conversation(tab_name)
            graph = conversation["graph"]

            with tab_container:
                tab_key = tab_name.replace(" ", "_")
                st.subheader(tab_name)
                status = "enabled" if conversation.get("faiss_enabled", True) else "disabled"
                active_files = list(conversation.get("vectorstores", {}).keys()) if conversation.get("faiss_enabled", True) else [d.split("]")[0][1:] for d in conversation.get("direct_docs", []) if doc.startswith("[") and "]" in doc]
                st.caption(f"FAISS retrieval is currently **{status}** for this chat.")
                if active_files:
                    st.caption(f"📚 Active files in context: {', '.join(active_files)}")
                else:
                    st.caption("📚 No files in context")

                uploaded_files = st.file_uploader(
                    "Upload PDF files",
                    type=["pdf"],
                    accept_multiple_files=True,
                    key=f"pdf_uploader_{tab_key}",
                    label_visibility="collapsed",
                )
                handle_pdf_upload(uploaded_files, conversation, tab_name)


                for idx, (role, msg) in enumerate(conversation["chat_history"]):
                    with st.chat_message(role):
                        st.markdown(msg)
                        
                        # Add "Fetch Better Results" button for assistant messages that might need better retrieval
                        if role == "assistant" and idx == len(conversation["chat_history"]) - 1:
                            # Check if message suggests nothing was found
                            not_found_keywords = [
                                "cannot find", "can't find", "not found", "no information",
                                "unable to find", "doesn't appear", "not in the", "not available"
                            ]
                            msg_lower = msg.lower()
                            if any(keyword in msg_lower for keyword in not_found_keywords):
                                if st.button("🔍 Fetch Better Results", key=f"enhanced_retry_{tab_key}_{idx}"):
                                    # Retry with enhanced retrieval
                                    last_user_msg = None
                                    for i in range(idx - 1, -1, -1):
                                        if conversation["chat_history"][i][0] == "user":
                                            last_user_msg = conversation["chat_history"][i][1]
                                            break
                                    
                                    if last_user_msg:
                                        # Remove the "not found" message and retry
                                        conversation["chat_history"] = conversation["chat_history"][:idx]
                                        st.session_state[f"enhanced_retry_{tab_key}"] = last_user_msg
                                        st.rerun()

                user_input = st.chat_input(
                    "Ask your question...", key=f"chat_input_{tab_key}"
                )
                
                if user_input:
                    # Add user message immediately and rerun to show it
                    conversation["chat_history"].append(("user", user_input))
                    # Store the input for processing after rerun
                    st.session_state[f"pending_input_{tab_key}"] = user_input
                    st.rerun()
                
                # Process enhanced retry if requested
                enhanced_retry_key = f"enhanced_retry_{tab_key}"
                is_enhanced = enhanced_retry_key in st.session_state
                if is_enhanced:
                    user_input = st.session_state.pop(enhanced_retry_key)
                elif f"pending_input_{tab_key}" in st.session_state:
                    user_input = st.session_state.pop(f"pending_input_{tab_key}")
                else:
                    user_input = None
                
                if user_input:
                    # Always fetch fresh context for every query (FAISS similarity search happens here)
                    context_blob = build_context_blob(conversation, user_input, enhanced=is_enhanced)
                    
                    messages = []
                    # Always include context if available - use SystemMessage for first message,
                    # append to user message for subsequent messages (LangGraph limitation)
                    if context_blob:
                        is_first_message = len(conversation["chat_history"]) == 1
                        if is_first_message:
                            messages.append(SystemMessage(
                                content=(
                                    "The user uploaded PDFs. Use the following context when relevant:\n"
                                    f"{context_blob}"
                                )
                            ))
                            messages.append(HumanMessage(content=user_input))
                        else:
                            # For subsequent messages, include context in user message
                            # This ensures FAISS vectors are fetched and included on every query
                            user_content = f"{user_input}\n\n[Context from uploaded PDFs - retrieved via FAISS similarity search]\n{context_blob}"
                            messages.append(HumanMessage(content=user_content))
                    else:
                        messages.append(HumanMessage(content=user_input))

                    config = {
                        "configurable": {
                            "thread_id": conversation["thread_id"],
                        }
                    }

                    st.session_state.active_tool_context = tab_name
                    try:
                        with st.spinner("Thinking..."):
                            start_time = time.time()
                            response = graph.invoke(
                                {"messages": messages}, config=config
                            )
                            response_time = time.time() - start_time
                        ai_message = next(
                            (
                                m.content.strip()
                                for m in reversed(response["messages"])
                                if isinstance(m, (AIMessage, ToolMessage)) and m.content
                            ),
                            None,
                        )
                        if ai_message:
                            # Format response time
                            if response_time < 60:
                                time_str = f"{response_time:.1f}s"
                            else:
                                minutes = int(response_time // 60)
                                seconds = int(response_time % 60)
                                time_str = f"{minutes}m {seconds}s"
                            
                            # Append response time to message
                            ai_message_with_time = f"{ai_message}\n\n*⏱️ Response time: {time_str}*"
                            conversation["chat_history"].append(("assistant", ai_message_with_time))
                        else:
                            error_msg = "⚠️ No response generated. Please try again."
                            st.warning(error_msg)
                            conversation["chat_history"].append(("assistant", error_msg))
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        conversation["chat_history"].append(("assistant", error_msg))
                    finally:
                        st.session_state.active_tool_context = None
                    
                    st.rerun()
