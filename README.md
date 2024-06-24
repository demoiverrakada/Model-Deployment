# Model Deployment

```
https://replicate.com/demoiverrakada/bert-base-uncased
```
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Setup Python, Docker and Cog in your local machine

To set Cog use below command
```
curl https://replicate.github.io/codespaces/scripts/install-cog.sh | bash
```

## Testing on Local Machine

### Create the Dockerfile

```
# Use the official PyTorch image from the Docker Hub
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]

```

### Create the requirements.txt

```
flask
torch
transformers
```

### Create the app.py

```
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
```

### Build the docker image and test it

```
docker build -t bert-app .
docker run -p 5000:5000 bert-app
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, world!"}'
```

## Deployment

### Create the cog.yaml file

```
# cog.yaml

build:
  python_version: "3.8"
  python_packages:
    - torch
    - transformers
    - flask

predict: "predict.py:Predictor"
```

### Create the predict.py file

```
# predict.py

from cog import BasePredictor, Input, Path
from transformers import BertTokenizer, BertModel
import torch

class Predictor(BasePredictor):
    def setup(self):
        """Load the pre-trained model and tokenizer"""
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)

    def predict(self, text: str = Input(description="Input text to generate embeddings")) -> list:
        """Generate embeddings for the input text"""
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze().tolist()
```

### Deploy the model on replicate using below commands

```
cog push r8.im/<your-username>/bert-base-uncased
```


