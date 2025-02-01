import streamlit as st
from vllm import LLM
from transformers import AutoTokenizer
import torch

# Load the model and tokenizer from Hugging Face
model_name = "akhilsheri57/DeepSeek-R1-Medical-COT"
model = LLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the prediction function
def predict(text: str):
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].cuda()  # Use GPU if available

    # Perform model inference
    with torch.no_grad():
        output = model.generate(input_ids=input_ids)

    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Streamlit UI
st.title("Medical Text Prediction App")
st.markdown("Enter some medical text, and the model will generate a response.")

# User input
user_input = st.text_area("Enter medical text here:")

if st.button("Generate Response"):
    if user_input:
        with st.spinner('Generating response...'):
            result = predict(user_input)
            st.success("Prediction Generated:")
            st.write(result)
    else:
        st.warning("Please enter some text to generate a prediction.")
