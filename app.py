import torch
from transformers import AutoTokenizer
from vllm import LLM

# Load the model and tokenizer
model_name = "akhilsheri57/DeepSeek-R1-Medical-COT"
model = LLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move the model to the device (CPU or GPU)

# Define the prediction function
def predict(text: str):
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)  # Ensure the tensor is on the correct device (CPU/GPU)

    # Perform model inference
    with torch.no_grad():
        output = model.generate(input_ids=input_ids)

    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
