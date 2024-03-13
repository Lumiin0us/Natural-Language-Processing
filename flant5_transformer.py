from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# Input text
sentence = 'alan Crane is the head of bar association in New Jersey, althought a good lawyer he is stubborn and annoying.'
input_text = f"Can you summarize this sentence, {sentence}"

# Tokenize input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate summary
output = model.generate(input_ids)

# Decode the generated summary
summary = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Summary:", summary)
