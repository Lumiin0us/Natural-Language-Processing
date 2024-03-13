import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

@st.cache(allow_output_mutation=True)
def load_model():
    """Load the tokenizer and model."""
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    return tokenizer, model

def generate_summary(text):
    """Generate summary for the input text."""
    tokenizer, model = load_model()
    input_text = f"Can you summarize this sentence, {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length = 200)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary

def main():
    """Streamlit UI layout and logic."""
    st.title("T5 Sentence Summarizer")
    user_input = st.text_area("Enter a sentence to summarize", "Type your text here...")
    if st.button("Summarize"):
        summary = generate_summary(user_input)
        st.write("Generated Summary:", summary)

if __name__ == "__main__":
    main()
