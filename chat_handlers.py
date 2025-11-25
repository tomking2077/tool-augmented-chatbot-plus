"""Chat rendering and handling logic"""
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from llm_setup import graph_builder
from document_handling import (
    build_context_blob,
    handle_pdf_upload,
)
from token_tracking import extract_token_usage
from ui_components import render_dual_chat_debug


def ensure_conversation(tab_name: str):
    """Ensure a conversation exists for a given tab name with all required fields."""
    conversations = st.session_state.conversations
    
    if tab_name not in conversations:
        checkpointer = MemorySaver()
        conversations[tab_name] = {
            "thread_id": f"user-{uuid.uuid4().hex[:8]}",
            "checkpointer": checkpointer,
            "chat_history": [],
            "graph": graph_builder.compile(checkpointer=checkpointer),
            "faiss_enabled": True,
            "token_stats": [],
            "vectorstores": {},
            "direct_docs": [],
            "last_upload_signature": None,
        }
    elif "graph" not in conversations[tab_name]:
        conversations[tab_name]["graph"] = graph_builder.compile(
            checkpointer=conversations[tab_name]["checkpointer"]
        )

    conversation = conversations[tab_name]
    conversation.setdefault("faiss_enabled", True)
    conversation.setdefault("token_stats", [])
    conversation.setdefault("vectorstores", {})
    conversation.setdefault("direct_docs", [])
    conversation.setdefault("last_upload_signature", None)
    
    return conversation


def _prepare_messages(conversation, user_input, enhanced=False):
    """Prepare messages for LLM with context from documents."""
    context_blob = build_context_blob(conversation, user_input, enhanced=enhanced)
    messages = []
    
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
            user_content = (
                f"{user_input}\n\n"
                "[Context from uploaded PDFs - retrieved via FAISS similarity search]\n"
                f"{context_blob}"
            )
            messages.append(HumanMessage(content=user_content))
    else:
        messages.append(HumanMessage(content=user_input))
    
    return messages
        

def _process_single_chat(chat_name, conversation, messages, user_input, shared_interaction):
    """Process a single chat request in a thread - returns result dict."""
    # Set active tool context for this thread to support tool calls
    if "active_tool_context" not in st.session_state:
         st.session_state.active_tool_context = chat_name
    
    config = {"configurable": {"thread_id": conversation["thread_id"]}}
    
    try:
        start_time = time.time()
        response = conversation["graph"].invoke({"messages": messages}, config=config)
        response_time = time.time() - start_time
        
        ai_message = next(
            (
                (m.content.strip() if isinstance(m.content, str) else " ".join(m.content).strip())
                for m in reversed(response["messages"])
                if isinstance(m, (AIMessage, ToolMessage)) and m.content
            ),
            None,
        )
        
        if ai_message:
            if response_time < 60:
                time_str = f"{response_time:.1f}s"
            else:
                minutes = int(response_time // 60)
                seconds = int(response_time % 60)
                time_str = f"{minutes}m {seconds}s"
            
            ai_message_with_time = f"{ai_message}\n\n*⏱️ Response time: {time_str}*"
            
            user_input_text = next(
                (m.content for m in messages if isinstance(m, HumanMessage)),
                user_input
            )
            token_data = extract_token_usage(response, user_input_text, ai_message)
            
            return {
                "chat_name": chat_name,
                "success": True,
                "message": ai_message_with_time,
                "token_data": token_data,
                "shared_interaction": shared_interaction
            }
        else:
            return {
                "chat_name": chat_name,
                "success": False,
                "message": "⚠️ No response generated. Please try again."
            }
    except Exception as e:
        return {
            "chat_name": chat_name,
            "success": False,
            "message": f"❌ Error: {str(e)}"
        }


def _handle_dual_chat_response(user_input, faiss_chat, direct_chat, status_placeholder, enhanced=False):
    """Handle user input and process responses from both chats asynchronously."""
    processing_key = "dual_chat_processing"
    
    # Check if processing is already in progress
    if processing_key not in st.session_state:
        if user_input is None:
            return  # No new input and no processing, nothing to do
        
        shared_interaction = max(
            len(faiss_chat.get("token_stats", [])),
            len(direct_chat.get("token_stats", []))
        ) + 1
        
        faiss_messages = _prepare_messages(faiss_chat, user_input, enhanced=enhanced)
        direct_messages = _prepare_messages(direct_chat, user_input, enhanced=enhanced)
        
        # Start both chats in parallel using ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=2)
        
        # Create tasks but attach script context manually
        # Note: add_script_run_ctx attaches context to current thread if called without args,
        # but here we need to attach to the new threads. 
        # The best way with ThreadPoolExecutor is to wrap the function.
        
        ctx = st.runtime.scriptrunner.get_script_run_ctx()
        
        def _task_wrapper(func, *args, **kwargs):
            add_script_run_ctx(ctx=ctx)
            return func(*args, **kwargs)
            
        st.session_state[processing_key] = {
            "executor": executor,
            "faiss_future": executor.submit(
                _task_wrapper,
                _process_single_chat,
                "FAISS Chat", faiss_chat, faiss_messages, user_input, shared_interaction
            ),
            "direct_future": executor.submit(
                _task_wrapper,
                _process_single_chat,
                "Direct Chat", direct_chat, direct_messages, user_input, shared_interaction
            ),
            "faiss_done": False,
            "direct_done": False
        }
    
    processing = st.session_state[processing_key]
    
    # Check FAISS result
    if not processing["faiss_done"] and processing["faiss_future"].done():
        try:
            result = processing["faiss_future"].result()
            faiss_chat["chat_history"].append(("assistant", result["message"]))
            
            if result["success"] and "token_data" in result:
                stats = faiss_chat.setdefault("token_stats", [])
                stats.append({
                    "interaction": result["shared_interaction"],
                    "input_tokens": result["token_data"].get("input", 0),
                    "output_tokens": result["token_data"].get("output", 0),
                    "total_tokens": result["token_data"].get("total", 0)
                })
                if len(stats) > 50:
                    faiss_chat["token_stats"] = stats[-50:]
            
            processing["faiss_done"] = True
            st.rerun()
        except Exception as e:
            faiss_chat["chat_history"].append(("assistant", f"❌ Error: {str(e)}"))
            processing["faiss_done"] = True
            st.rerun()
    
    # Check Direct result
    if not processing["direct_done"] and processing["direct_future"].done():
        try:
            result = processing["direct_future"].result()
            direct_chat["chat_history"].append(("assistant", result["message"]))
            
            if result["success"] and "token_data" in result:
                stats = direct_chat.setdefault("token_stats", [])
                stats.append({
                    "interaction": result["shared_interaction"],
                    "input_tokens": result["token_data"].get("input", 0),
                    "output_tokens": result["token_data"].get("output", 0),
                    "total_tokens": result["token_data"].get("total", 0)
                })
                if len(stats) > 50:
                    direct_chat["token_stats"] = stats[-50:]
            
            processing["direct_done"] = True
            st.rerun()
        except Exception as e:
            direct_chat["chat_history"].append(("assistant", f"❌ Error: {str(e)}"))
            processing["direct_done"] = True
            st.rerun()
    
    # Update status
    if not processing["faiss_done"] or not processing["direct_done"]:
        remaining = []
        if not processing["faiss_done"]:
            remaining.append("FAISS")
        if not processing["direct_done"]:
            remaining.append("Direct")
        
        with status_placeholder.container():
            if len(remaining) == 2:
                st.info("🔄 Processing in both chats...")
            else:
                st.info(f"🔄 Waiting for: {', '.join(remaining)} chat...")
        
        time.sleep(0.1)
        st.rerun()
    else:
        # Both done - cleanup
        processing["executor"].shutdown(wait=False)
        del st.session_state[processing_key]
        status_placeholder.empty()


def render_dual_chat_mode():
    """Render side-by-side dual chat mode for FAISS comparison."""
    faiss_chat = ensure_conversation("FAISS Chat")
    direct_chat = ensure_conversation("Direct Chat")
    
    # Set FAISS settings
    faiss_chat["faiss_enabled"] = True
    direct_chat["faiss_enabled"] = False
    
    # PDF uploader in sidebar
    with st.sidebar:
        st.markdown("### 📄 PDF Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="dual_chat_pdf_uploader",
            label_visibility="visible",
        )
        if uploaded_files:
            st.caption(f"📎 {len(uploaded_files)} file(s) uploaded")
    
    # Handle PDF uploads
    if uploaded_files:
        # Use the consolidated handler from document_handling
        handle_pdf_upload(uploaded_files, faiss_chat, "FAISS Chat")
        handle_pdf_upload(uploaded_files, direct_chat, "Direct Chat")
    
    # Shared debug panel
    render_dual_chat_debug(faiss_chat, direct_chat)
    
    # Status placeholder for processing messages
    status_placeholder = st.empty()
    
    # Two columns for side-by-side chats with separator
    st.markdown(
        """
        <style>
        [data-testid="column"]:first-child {
            border-right: 2px solid #e0e0e0;
            padding-right: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2)
    
    # FAISS Chat column
    with col1:
        st.subheader("🔍 FAISS Chat")
        st.caption("Using vector search retrieval")
        
        for idx, (role, msg) in enumerate(faiss_chat["chat_history"]):
            with st.chat_message(role):
                st.markdown(msg)
                
                # Add "Fetch Better Results" button for assistant messages
                if role == "assistant" and idx == len(faiss_chat["chat_history"]) - 1:
                    not_found_keywords = [
                        "cannot find", "can't find", "not found", "no information",
                        "unable to find", "doesn't appear", "not in the", "not available"
                    ]
                    msg_lower = msg.lower()
                    if any(keyword in msg_lower for keyword in not_found_keywords):
                        if st.button("🔍 Fetch Better Results", key=f"enhanced_faiss_{idx}"):
                            last_user_msg = None
                            for i in range(idx - 1, -1, -1):
                                if faiss_chat["chat_history"][i][0] == "user":
                                    last_user_msg = faiss_chat["chat_history"][i][1]
                                    break
                            if last_user_msg:
                                faiss_chat["chat_history"] = faiss_chat["chat_history"][:idx]
                                st.session_state["enhanced_retry_faiss"] = last_user_msg
                                st.rerun()
        
        # Show thinking indicator if processing
        processing_key = "dual_chat_processing"
        if processing_key in st.session_state:
            processing = st.session_state[processing_key]
            if not processing.get("faiss_done", False):
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        st.markdown("_Processing your question..._")
    
    # Direct Chat column
    with col2:
        st.subheader("📄 Direct Chat")
        st.caption("Using direct document text")
        
        for idx, (role, msg) in enumerate(direct_chat["chat_history"]):
            with st.chat_message(role):
                st.markdown(msg)
                
                # Add "Fetch Better Results" button for assistant messages
                if role == "assistant" and idx == len(direct_chat["chat_history"]) - 1:
                    not_found_keywords = [
                        "cannot find", "can't find", "not found", "no information",
                        "unable to find", "doesn't appear", "not in the", "not available"
                    ]
                    msg_lower = msg.lower()
                    if any(keyword in msg_lower for keyword in not_found_keywords):
                        if st.button("🔍 Fetch Better Results", key=f"enhanced_direct_{idx}"):
                            last_user_msg = None
                            for i in range(idx - 1, -1, -1):
                                if direct_chat["chat_history"][i][0] == "user":
                                    last_user_msg = direct_chat["chat_history"][i][1]
                                    break
                            if last_user_msg:
                                direct_chat["chat_history"] = direct_chat["chat_history"][:idx]
                                st.session_state["enhanced_retry_direct"] = last_user_msg
                                st.rerun()
        
        # Show thinking indicator if processing
        if processing_key in st.session_state:
            processing = st.session_state[processing_key]
            if not processing.get("direct_done", False):
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        st.markdown("_Processing your question..._")
        
    
    # Shared input at bottom
    user_input = st.chat_input(
        "Ask your question (will be sent to both chats)...",
        key="dual_chat_input"
    )
    
    if user_input:
        # Add user message to both chats immediately and rerun to show it
        faiss_chat["chat_history"].append(("user", user_input))
        direct_chat["chat_history"].append(("user", user_input))
        st.session_state["pending_dual_chat_input"] = user_input
        
        # Show confirmation that message was sent
        with status_placeholder.container():
            st.success("✅ Message sent! Processing in both chats...")
        
        st.rerun()

    # Process enhanced retry for FAISS
    if "enhanced_retry_faiss" in st.session_state:
        user_input = st.session_state.pop("enhanced_retry_faiss")
        faiss_chat["chat_history"].append(("user", user_input))
        _handle_dual_chat_response(user_input, faiss_chat, direct_chat, status_placeholder, enhanced=True)
    # Process enhanced retry for Direct
    elif "enhanced_retry_direct" in st.session_state:
        user_input = st.session_state.pop("enhanced_retry_direct")
        direct_chat["chat_history"].append(("user", user_input))
        _handle_dual_chat_response(user_input, faiss_chat, direct_chat, status_placeholder, enhanced=True)
    # Process pending input (from previous rerun) or check ongoing processing
    elif "pending_dual_chat_input" in st.session_state:
        user_input = st.session_state.pop("pending_dual_chat_input")
        _handle_dual_chat_response(user_input, faiss_chat, direct_chat, status_placeholder)
    elif "dual_chat_processing" in st.session_state:
        # Check ongoing processing even if no new input
        _handle_dual_chat_response(None, faiss_chat, direct_chat, status_placeholder)
