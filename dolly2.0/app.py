from flask import Flask, render_template, request, jsonify
import torch
from transformers import pipeline

app = Flask(__name__)

# Load Dolly-v2-12b model
generate_text = pipeline(
    model="databricks/dolly-v2-12b",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

# Home page with the input form
@app.route("/")
def home():
    return render_template("index.html")

# Generate content ideas
@app.route("/generate_ideas", methods=["POST"])
def generate_ideas():
    keyword = request.form["keyword"]
    prompt = f"Generate 5 content ideas related to {keyword}"
    ideas = generate_text(prompt, max_length=150)
    return jsonify(ideas)

# Optimize title
@app.route("/optimize_title", methods=["POST"])
def optimize_title():
    title = request.form["title"]
    prompt = f"Optimize the following title: {title}"
    optimized_title = generate_text(prompt, max_length=150, num_return_sequences=1)
    return jsonify(optimized_title)

if __name__ == "__main__":
    app.run(debug=True)
