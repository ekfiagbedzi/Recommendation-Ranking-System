from utils.helpers import text_processor
from transformers import BertModel, BertTokenizer

sentence="Go away and come another day"
model=BertModel.from_pretrained("bert-base-uncased")
tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")

if __name__ == "__main__":
    embeddings = text_processor(sentence=sentence, model=model, tokenizer=tokenizer, max_length=50)
    print(embeddings, embeddings.shape)