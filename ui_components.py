"""UI components: debug panels, charts, and visualizations"""
import pandas as pd
import streamlit as st


def render_dual_chat_debug(faiss_chat: dict, direct_chat: dict):
    """Render shared debug panel for dual-chat mode comparing FAISS vs Direct"""
    with st.expander("Debug & Token Usage Comparison", expanded=False):
        st.caption("Token usage comparison: FAISS vs Direct")
        
        faiss_stats = faiss_chat.get("token_stats", [])
        direct_stats = direct_chat.get("token_stats", [])
        
        if faiss_stats or direct_stats:
            # Combine stats from both chats
            all_stats = []
            for stat in faiss_stats:
                all_stats.append({**stat, "method": "FAISS"})
            for stat in direct_stats:
                all_stats.append({**stat, "method": "Direct"})
            
            if all_stats:
                df = pd.DataFrame(all_stats)
                
                # Calculate total input/output/total tokens (sums, not averages)
                faiss_total_input = df[df["method"] == "FAISS"]["input_tokens"].sum() if "FAISS" in df["method"].values else 0
                faiss_total_output = df[df["method"] == "FAISS"]["output_tokens"].sum() if "FAISS" in df["method"].values else 0
                faiss_total_combined = faiss_total_input + faiss_total_output
                
                direct_total_input = df[df["method"] == "Direct"]["input_tokens"].sum() if "Direct" in df["method"].values else 0
                direct_total_output = df[df["method"] == "Direct"]["output_tokens"].sum() if "Direct" in df["method"].values else 0
                direct_total_combined = direct_total_input + direct_total_output
                
                savings = max(direct_total_combined - faiss_total_combined, 0) if direct_total_combined > 0 else 0
                percent_saved = (savings / direct_total_combined * 100) if direct_total_combined > 0 else 0
                
                # Create layout with chart on left, metrics on right
                chart_col, metrics_col = st.columns([0.65, 0.35])
                
                with chart_col:
                    # Prepare data for Streamlit line chart
                    # Create a wide-format dataframe with all token metrics
                    interactions = sorted(df["interaction"].unique())
                    chart_data = []
                    
                    # Start at interaction 0 with all values at 0
                    chart_data.append({
                        "Interaction": 0,
                        "FAISS Input": 0,
                        "FAISS Output": 0,
                        "FAISS Total": 0,
                        "Direct Input": 0,
                        "Direct Output": 0,
                        "Direct Total": 0,
                    })
                    
                    for i in interactions:
                        faiss_row = df[(df["interaction"] == i) & (df["method"] == "FAISS")]
                        direct_row = df[(df["interaction"] == i) & (df["method"] == "Direct")]
                        
                        chart_data.append({
                            "Interaction": i,
                            "FAISS Input": faiss_row["input_tokens"].iloc[0] if len(faiss_row) > 0 else 0,
                            "FAISS Output": faiss_row["output_tokens"].iloc[0] if len(faiss_row) > 0 else 0,
                            "FAISS Total": faiss_row["total_tokens"].iloc[0] if len(faiss_row) > 0 else 0,
                            "Direct Input": direct_row["input_tokens"].iloc[0] if len(direct_row) > 0 else 0,
                            "Direct Output": direct_row["output_tokens"].iloc[0] if len(direct_row) > 0 else 0,
                            "Direct Total": direct_row["total_tokens"].iloc[0] if len(direct_row) > 0 else 0,
                        })
                    
                    chart_df = pd.DataFrame(chart_data).set_index("Interaction")
                    
                    # Use st.line_chart to show all metrics
                    # Color list matches the order of columns: FAISS Input, FAISS Output, FAISS Total, Direct Input, Direct Output, Direct Total
                    # Green shades (medium-light to dark): FAISS Input, FAISS Output, FAISS Total
                    # Red shades (medium-light to dark): Direct Input, Direct Output, Direct Total
                    # Make graph 25% taller: 350 * 1.25 = 437.5, round to 438
                    st.line_chart(
                        chart_df,
                        height=438,
                        color=["#bae4b3", "#74c476", "#238b45", "#fcae91", "#fb6a4a", "#cb181d"]
                    )
                
                with metrics_col:
                    st.markdown("### Metrics")
                    if faiss_total_combined > 0 and direct_total_combined > 0:
                        # Show FAISS and Direct side by side
                        faiss_col, direct_col = st.columns(2)
                        
                        with faiss_col:
                            st.markdown("**FAISS**")
                            st.metric("Input", f"{faiss_total_input:.0f}")
                            st.metric("Output", f"{faiss_total_output:.0f}")
                            st.metric("Total", f"{faiss_total_combined:.0f}")
                        
                        with direct_col:
                            st.markdown("**Direct**")
                            st.metric("Input", f"{direct_total_input:.0f}")
                            st.metric("Output", f"{direct_total_output:.0f}")
                            st.metric("Total", f"{direct_total_combined:.0f}")
                        
                        st.markdown("---")
                        st.metric("Tokens Saved", f"{savings:.0f}", delta=f"{percent_saved:.1f}%")
                    elif faiss_total_combined > 0:
                        st.markdown("**FAISS**")
                        st.metric("Input", f"{faiss_total_input:.0f}")
                        st.metric("Output", f"{faiss_total_output:.0f}")
                        st.metric("Total", f"{faiss_total_combined:.0f}")
                    elif direct_total_combined > 0:
                        st.markdown("**Direct**")
                        st.metric("Input", f"{direct_total_input:.0f}")
                        st.metric("Output", f"{direct_total_output:.0f}")
                        st.metric("Total", f"{direct_total_combined:.0f}")
            else:
                st.info("No token data logged yet.")
        else:
            st.info("No token data logged yet. Ask questions to see comparison.")
