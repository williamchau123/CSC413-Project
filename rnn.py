import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import concatenate_datasets, load_dataset
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import evaluate

df = load_dataset("csv", data_files="data.csv")
df = df.remove_columns(['ID', 'Source'])
redit = df["train"].select(range(125)).shuffle(seed=42)
stack = df["train"].select(range(125, 250)).shuffle(seed=42)
gpt_redit = df["train"].select(range(250, 375)).shuffle(seed=42)
gpt_stack = df["train"].select(range(375, 500)).shuffle(seed=42)

gpt_df2 = load_dataset("csv", data_files="gpt.csv")
human_df2 = load_dataset("csv", data_files="human.csv")
gpt_df2 = gpt_df2.remove_columns(['ID', 'Source'])
human_df2 = human_df2.remove_columns(['ID', 'Source'])
gpt_df2 = gpt_df2["train"].shuffle(seed=42)
human_df2 = human_df2["train"].shuffle(seed=42)
gpt_df2 = gpt_df2.select(i for i in range(len(gpt_df2)) if gpt_df2["text"][i] is not None)
human_df2 = human_df2.select(i for i in range(len(human_df2)) if human_df2["text"][i] is not None)
gpt_df2 = gpt_df2.select(range(250))
human_df2 = human_df2.select(range(250))

tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
# for internet connection error, please download the model from https://huggingface.co/bert-base-cased and place the
# folder in the same directory model = AutoModelForSequenceClassification.from_pretrained("./bert-base-cased",
# num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_redit = redit.map(tokenize_function, batched=True)
tokenized_stack = stack.map(tokenize_function, batched=True)
tokenized_gpt_redit = gpt_redit.map(tokenize_function, batched=True)
tokenized_gpt_stack = gpt_stack.map(tokenize_function, batched=True)
tokenized_gpt_stack2 = gpt_df2.map(tokenize_function, batched=True)
tokenized_human_stack2 = human_df2.map(tokenize_function, batched=True)
train_set = concatenate_datasets([tokenized_redit.select(range(50)), tokenized_stack.select(range(50)),
                                  tokenized_gpt_redit.select(range(50)), tokenized_gpt_stack.select(range(50)),
                                  tokenized_gpt_stack2.select(range(100)), tokenized_human_stack2.select(range(100))]).shuffle(seed=42)
val_set = concatenate_datasets([tokenized_redit.select(range(50, 100)), tokenized_stack.select(range(50, 100)),
                                tokenized_gpt_redit.select(range(50, 100)), tokenized_gpt_stack.select(range(50, 100)),
                                tokenized_gpt_stack2.select(range(100, 200)), tokenized_human_stack2.select(range(100, 200))]).shuffle(seed=42)
test_set = concatenate_datasets([tokenized_redit.select(range(100, 125)), tokenized_stack.select(range(100, 125)),
                                 tokenized_gpt_redit.select(range(100, 125)), tokenized_gpt_stack.select(range(100, 125)),
                                 tokenized_gpt_stack2.select(range(200, 250)), tokenized_human_stack2.select(range(200, 250))]).shuffle(seed=42)

args = TrainingArguments(
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-7,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True,
    output_dir="test_trainer",
    metric_for_best_model="accuracy",
    report_to="none",
    # resume_from_checkpoint=True
)

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

def plot():
    print(trainer.state.log_history)
    losses, iters, acc = [], [], []
    losses_train, epochs = [], []
    for i, epoch in enumerate(trainer.state.log_history[:-1]):
        if 'loss' in epoch:
            print(epoch)
            losses_train.append(epoch["loss"])
            epochs.append(epoch["epoch"])
        elif 'eval_loss' in epoch:
            losses.append(epoch['eval_loss'])
            iters.append(epoch['epoch'])
            acc.append(epoch['eval_accuracy'])

    plt.title("Train Learning Curve")
    plt.plot(epochs, losses_train, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Valid Learning Curve")
    plt.plot(iters, losses, label="Valid Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Valid Learning Curve")
    plt.plot(iters, acc, label="Valid Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

train = trainer.train()
evaluate = trainer.evaluate()
print(train)
print(evaluate)
plot()
# print(test_set.select(range(2)))
pred = trainer.predict(test_set)
print(pred)
print(pred.metrics)
