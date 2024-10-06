import streamlit as st
import requests
import json
import base64
import io

def streamlit_app():
    st.title("RAG Comparison: Standard vs Agentic")
    question = st.text_input("Enter your question:", "Who is Heinrich Rudolph Hertz?")
    if st.button("Submit"):
        with st.spinner("Processing..."):
            try:
                response = requests.post("http://localhost:8000/query", json={"question": question})
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as e:
                st.error(f"Error communicating with the server: {str(e)}")
                return
            except json.JSONDecodeError:
                st.error("Error decoding server response")
                return
        
        # Display results
        for rag_type in ["standard_rag", "agentic_rag"]:
            st.subheader(f"{rag_type.replace('_', ' ').title()}")
            st.write(f"Answer: {result[rag_type]['response']['answer']}")
            st.write(f"Time taken: {result[rag_type]['time']:.6f} seconds")
            st.write("Sources:")
            for source in result[rag_type]['response']['sources']:
                st.write(f"- {source}")
            
            if 'reflection' in result[rag_type]['response']:
                st.write(f"Reflection: {result[rag_type]['response']['reflection']}")
        
        # Handle audio for standard_rag
        if 'speech' in result['standard_rag']:
            try:
                audio_base64 = result['standard_rag']['speech']
                audio_bytes = base64.b64decode(audio_base64)
                audio_io = io.BytesIO(audio_bytes)
                st.audio(audio_io, format='audio/wav')
                st.success("Audio generated successfully!")
            except Exception as e:
                st.error(f"Error processing audio data: {str(e)}")
        else:
            st.warning("No speech data found for the Standard RAG answer.")
        
        st.write("Debug Information:")
        st.json(result)

if __name__ == "__main__":
    streamlit_app()