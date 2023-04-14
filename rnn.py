import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate

df = load_dataset("csv", data_files="data.csv")


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# for internet connection error, please download the model from https://huggingface.co/bert-base-cased and place the folder in the same directory
# model = AutoModelForSequenceClassification.from_pretrained("./bert-base-cased", num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = df.map(tokenize_function, batched=True)
train_set = tokenized_datasets['train'].select(range(200))
val_set = tokenized_datasets['train'].select(range(200, 400))
test_set = tokenized_datasets['train'].select(range(400, 500))

args = TrainingArguments(
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    output_dir="test_trainer",
    metric_for_best_model="accuracy"
)
training_args = TrainingArguments(output_dir="test_trainer")

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
    
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=val_set,
    compute_metrics=compute_metrics,
)
train = trainer.train()
evaluate = trainer.evaluate()
print(train)
print(evaluate)