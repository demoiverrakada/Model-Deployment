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
