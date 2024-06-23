from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('text', '')

    # Encode text
    inputs = tokenizer(input_text, return_tensors='pt')

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert outputs to list for JSON serialization
    outputs = outputs.last_hidden_state.squeeze().tolist()

    return jsonify(outputs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
