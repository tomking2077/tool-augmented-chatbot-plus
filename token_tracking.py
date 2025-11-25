"""Token usage tracking and extraction"""
import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage


def extract_token_usage(response_payload, user_text: str, ai_text: str) -> dict:
    """Extract input, output, and total tokens from response. Returns dict with input, output, total."""
    # Fallback word count estimates
    input_estimate = len(user_text.split())
    output_estimate = len(ai_text.split())
    total_estimate = input_estimate + output_estimate
    
    if not response_payload:
        return {"input": input_estimate, "output": output_estimate, "total": total_estimate}
    
    for message in reversed(response_payload.get("messages", [])):
        usage = getattr(message, "usage_metadata", None)
        if usage:
            # Try to get detailed breakdown
            prompt_tokens = usage.get("prompt_token_count") or usage.get("input_tokens")
            candidates_tokens = usage.get("candidates_token_count") or usage.get("output_tokens")
            total_tokens = usage.get("total_token_count") or usage.get("total_tokens")
            
            if total_tokens is not None:
                return {
                    "input": prompt_tokens if prompt_tokens is not None else (total_tokens - (candidates_tokens or 0)),
                    "output": candidates_tokens if candidates_tokens is not None else 0,
                    "total": total_tokens
                }
        
        metadata = getattr(message, "response_metadata", None)
        if metadata:
            token_count = metadata.get("token_count") or metadata.get("usage")
            if isinstance(token_count, dict):
                total = token_count.get("total_tokens") or token_count.get("total_token_count")
                prompt = token_count.get("prompt_token_count") or token_count.get("input_tokens")
                candidates = token_count.get("candidates_token_count") or token_count.get("output_tokens")
                
                if total is not None:
                    return {
                        "input": prompt if prompt is not None else (total - (candidates or 0)),
                        "output": candidates if candidates is not None else 0,
                        "total": total
                    }
    
    return {"input": input_estimate, "output": output_estimate, "total": total_estimate}


def log_token_stats(tab_name: str, token_data: dict):
    """Log token statistics with input, output, and total breakdown"""
    conversation = st.session_state.conversations.get(tab_name)
    if not conversation:
        return
    stats = conversation.setdefault("token_stats", [])
    stats.append({
        "interaction": len(stats) + 1,
        "input_tokens": token_data.get("input", 0),
        "output_tokens": token_data.get("output", 0),
        "total_tokens": token_data.get("total", 0)
    })
    if len(stats) > 50:
        conversation["token_stats"] = stats[-50:]

