import transformers

# this is the maximum number of tokens in a sentence
MAX_LEN = 512

# batch sizes is small because model is huge!
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

# let's train for a maximum of 10 epochs
EPOCHS = 10

# define path to BERT model files
BERT_PATH = 'bert-base-uncased'

# this is where you want to save the model
MODEL_PATH = 'model.bin'

# define the tokenizer
# we use tokenizer and model
# from huggingface's transformers
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)