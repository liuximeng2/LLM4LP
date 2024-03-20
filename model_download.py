from transformers.models.auto import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers import LongformerConfig, LongformerModel
from transformers import LongformerTokenizer


print("downloading model")
configuration = LongformerConfig()
model = LongformerModel(configuration)
model.save_pretrained("./model/pre_train/longformer")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
tokenizer.save_pretrained("./model/tokenizer/longformer")
print(model)
"""
print("downloading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", use_fast = False)
tokenizer.save_pretrained("./model/pre_train/deberta-base")
model.save_pretrained("./model/tokenizer/deberta-base")

from transformers import DistilBertModel, DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.save_pretrained("./model/tokenizer/distilbert-base-uncased")

model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model.save_pretrained("./model/pre_train/distilbert-base-uncased")
"""