import streamlit as st
import sys

st.set_page_config(page_title="VayuVidh", page_icon="🌱")
st.title("VayuVidh 🌱")
st.caption("CBG Knowledge Assistant")

query = st.text_input("Ask a CBG question:")

if st.button("Ask"):
    if query:
        try:
            with st.spinner("Searching knowledge base..."):
                from chatbot import ask_question
                answer, sources = ask_question(query)
            
            st.markdown("### Answer")
            st.write(answer)
            
            st.markdown("### Sources")
            for i, s in enumerate(sources, 1):
                st.write(f"**{i}.** 📄 `{s['source_file']}` — Page {s['page']} | `{s['source_folder']}`")
        
        except Exception as e:
            st.error(f"ERROR: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning("Please type a question first.")