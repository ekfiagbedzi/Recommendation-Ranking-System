from sklearn.metrics import max_error
import torch
from transformers import BertTokenizer
from transformers import BertModel


model = BertModel.from_pretrained(
    "bert-base-uncased", output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

example_text = "If it quacks like a duck, it is probably a duck."
bert_input = tokenizer(
    example_text, padding="max_length", max_length=20,
    truncation=True, return_tensors="pt")

print(bert_input["input_ids"])

print(bert_input["token_type_ids"])

example_text = tokenizer.decode(bert_input.input_ids[0])
print(example_text)


model.eval()
sentences = [ 
              ['Never say that a duck cannot quack'],
              ['Gonna quack like a duck'],
              ["Give me your best quack"],
              ["You quack like a nice duck"],
              ["Up there, you quack like a duck"],
              ["Never try to quack like a duck"],
              ["Gonna make you quack like a duck"],
              ["Let me quack like a duck"],
              ["You got me quacking like a duck"],
              ["Down to quack city, where the quack is green and the ducks are pretty"],
            ]

for sentence in sentences:
    encoded = tokenizer.batch_encode_plus(sentence, max_length=15, padding="max_length", truncation=True)
    print(encoded)
    encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
    print(encoded)
    with torch.no_grad():
        outputs = model(**encoded)
        print(outputs.last_hidden_state.size())
print(outputs.last_hidden_state)
