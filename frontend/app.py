import streamlit as st
from services.backend_client import solve_problem, send_feedback

st.set_page_config(
    page_title="Math Mentor AI",
    layout="wide"
)

st.title("🧠 Math Mentor AI")
st.write("Reliable multi-agent math solver with explanations")

# -----------------------------
# INPUT
# -----------------------------
st.subheader("Enter your math problem")

user_input = st.text_area(
    "Type a math question (JEE-style)",
    height=120,
    placeholder="Example: Solve x squared minus 5x plus 6 equals zero"
)

solve_clicked = st.button("Solve")

# -----------------------------
# OUTPUT
# -----------------------------
if solve_clicked and user_input.strip():
    with st.spinner("Solving with AI agents..."):
        try:
            result = solve_problem(user_input)
        except Exception as e:
            st.error(f"Backend error: {e}")
            st.stop()

    st.success("Solved successfully")

    # Final Answer
    st.subheader("✅ Final Answer")
    st.write(result["final_answer"])

    # Explanation
    st.subheader("📘 Step-by-Step Explanation")
    for step in result["explanation"]:
        st.markdown(f"- {step}")

    # Confidence
    st.subheader("📊 Confidence")
    st.progress(min(int(result["confidence"] * 100), 100))
    st.write(f"{int(result['confidence'] * 100)}%")

    # Agent Trace
    st.subheader("🧩 Agent Trace")
    st.write(" → ".join(result["agent_trace"]))

    # Retrieved Context (RAG)
    if result["retrieved_context"]:
        st.subheader("📚 Retrieved Knowledge")
        for ctx in result["retrieved_context"]:
            st.markdown(f"**Source:** {ctx['source']}")
            st.markdown(ctx["content"])
            st.caption(f"Score: {ctx['score']:.2f}")

    # HITL Feedback
    st.subheader("🧑‍⚖️ Was this solution correct?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes"):
            send_feedback(True, "User confirmed correct")
            st.success("Thanks for your feedback!")

    with col2:
        if st.button("❌ No"):
            comment = st.text_input("What was wrong?")
            if comment:
                send_feedback(False, comment)
                st.warning("Feedback recorded for improvement")
