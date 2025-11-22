import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
st.header("ChatCut")
model_name = "Sachin-0001/dialogsum-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

st.title("Summarize Text")

text = st.text_area("Enter Text:",height = 250)

if st.button("Summarize"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    output = model.generate(**inputs, max_length=60)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write("### Summary:")
    st.write(summary)
