import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import streamlit as st

# Use CPU
device = torch.device("cpu")

# Set page config
st.set_page_config(page_title="Python Tutor", page_icon="🐍")
st.title("Python Tutor for Beginners (LoRA)")
st.markdown("Ask me any *Python programming* question below:")

# Load tokenizer and base model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0")
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    base_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlaMA/TinyLlaMA-1.1B-Chat-v1.0",
        torch_dtype=torch.float32,
    ).to(device)

    model = PeftModel.from_pretrained(base_model, "lora_adapter").to(device)
    model.eval()

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

    return tokenizer, model, classifier

tokenizer, model, classifier = load_model()

# User input
user_question = st.text_input("Your question:", placeholder="e.g. What is a Python dictionary?")

# Helper to check topic relevance
def is_python_related(question):
    candidate_labels = [
        "Python programming", "Java programming", "General knowledge",
        "Geography", "Politics", "Entertainment", "History", "Science"
    ]
    result = classifier(question, candidate_labels)
    return result['labels'][0] == "Python programming"

# Code block formatter
def format_code_blocks(text):
    if "```" in text:
        return text
    lines = text.split("\n")
    in_code = False
    formatted = []
    for line in lines:
        if line.strip().startswith((">>>", "#")) or line.strip().endswith(":") or ("=" in line and not line.strip().startswith("-")):
            if not in_code:
                formatted.append("```python")
                in_code = True
        elif in_code and line.strip() == "":
            formatted.append("```")
            in_code = False
        formatted.append(line)
    if in_code:
        formatted.append("```")
    return "\n".join(formatted)

# Process the question
if user_question:
    if len(user_question.strip()) < 10:
        st.warning("Please ask a more specific Python question.")
    else:
            if not is_python_related(user_question):
                st.error("Sorry, I am a Python tutor. I cannot answer this.")
            else:
                with st.spinner("Thinking..."):

                    prompt = f"""You are a helpful and knowledgeable Python programming tutor.
Always provide short, clear, beginner-friendly explanations with examples.

Question: {user_question}
Answer:"""

                    inputs = tokenizer(prompt, return_tensors="pt").to(device)

                    with torch.inference_mode():
                        output = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=False,
                            repetition_penalty=1.1,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id
                        )

                    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
                    answer = decoded.split("Answer:")[-1].split("Question:")[0].strip()  # clean hallucinated extra question
                    formatted = format_code_blocks(answer)

                    st.markdown("### 💡 Answer:")
                    with st.expander("🔍 Click to view response"):
                        st.markdown(formatted)
