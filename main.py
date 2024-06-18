# Advanced Natural Language Processing     Exercise 1
# Dvir Ben Asuli                           318208816
# The Hebrew University of Jerusalem       June 2024

import os
import json
import torch
import matplotlib
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, \
    TrainingArguments, get_scheduler, AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_dataset('glue', 'mrpc')


def plot_accuracy_vs_r_factor(r_factors, accuracy_scores, save_path):
    plt.plot(r_factors, accuracy_scores, marker='o')
    plt.xlabel('r Factor')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. r Factor')
    plt.grid(True)
    plt.savefig(save_path)  # Save the plot as an image file
    plt.close()


def train_and_evaluate_q1(model, train_loader, val_loader, num_epochs=3, learning_rate=3e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training and evaluation loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        print(f"Train loss for epoch {epoch + 1}: {sum(train_losses) / len(train_losses):.2f}")

        # Evaluate the model's accuracy
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch + 1}/{num_epochs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_losses.append(loss.item())

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            print(f"Validation loss for epoch {epoch + 1}: {sum(val_losses) / len(val_losses):.2f}")
            print(f"Validation accuracy for epoch {epoch + 1}: {accuracy * 100:.2f}%")

    # Evaluate the model
    results = {
        "num of Epochs": num_epochs,
        "Learning Rate": learning_rate,
        "Loss": sum(val_losses) / len(val_losses),
        "Accuracy": f"{accuracy * 100}%"
    }

    # Save results to a text file
    if not os.path.exists('./Q1_Results/'):
        os.makedirs('./Q1_Results/')
    with open('./Q1_Results/evaluation_results.txt', 'w') as f:
        f.write(json.dumps(results, indent=4))


def train_and_evaluate_q2(model, train_loader, val_loader, num_epochs=3, learning_rate=3e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        print(f"Train loss for epoch {epoch + 1}: {sum(train_losses) / len(train_losses):.2f}")

        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch + 1}/{num_epochs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_losses.append(loss.item())

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            print(f"Validation loss for epoch {epoch + 1}: {sum(val_losses) / len(val_losses):.2f}")
            print(f"Validation accuracy for epoch {epoch + 1}: {accuracy * 100:.2f}%")

    results = {
        "num of Epochs": num_epochs,
        "Learning Rate": learning_rate,
        "Loss": sum(val_losses) / len(val_losses),
        "Accuracy": f"{accuracy * 100}%"
    }

    if not os.path.exists('./Q2_Results/'):
        os.makedirs('./Q2_Results/')
    with open('./Q2_Results/evaluation_results32.txt', 'w') as f:
        f.write(json.dumps(results, indent=4))


def train_and_evaluate_q3a(model, train_loader, val_loader, num_epochs=3, learning_rate=3e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        print(f"Train loss for epoch {epoch + 1}: {sum(train_losses) / len(train_losses):.2f}")

        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch + 1}/{num_epochs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_losses.append(loss.item())

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            print(f"Validation loss for epoch {epoch + 1}: {sum(val_losses) / len(val_losses):.2f}")
            print(f"Validation accuracy for epoch {epoch + 1}: {accuracy * 100:.2f}%")

    results = {
        "num of Epochs": num_epochs,
        "Learning Rate": learning_rate,
        "Loss": sum(val_losses) / len(val_losses),
        "Accuracy": f"{accuracy * 100}%"
    }

    if not os.path.exists('./Q3_Results/'):
        os.makedirs('./Q3_Results/')
    with open('./Q3_Results/evaluation_results32.txt', 'w') as f:
        f.write(json.dumps(results, indent=4))


def train_and_evaluate_q3b(model, train_loader, val_loader, num_epochs=3, learning_rate=3e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f"Train loss for epoch {epoch + 1}: {sum(train_losses) / len(train_losses):.2f}")

        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch + 1}/{num_epochs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_losses.append(loss.item())

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            print(f"Validation loss for epoch {epoch + 1}: {sum(val_losses) / len(val_losses):.2f}")
            print(f"Validation accuracy for epoch {epoch + 1}: {accuracy * 100:.2f}%")

    results = {
        "num of Epochs": num_epochs,
        "Learning Rate": learning_rate,
        "Loss": sum(val_losses) / len(val_losses),
        "Accuracy": f"{accuracy * 100}%"
    }

    if not os.path.exists('./Q3_Results/'):
        os.makedirs('./Q3_Results/')
    with open('./Q3_Results/evaluation_results2_4.txt', 'w') as f:
        f.write(json.dumps(results, indent=4))


# Full Fine-tune
def q1():
    # Tokenize dataset to convert input text into numerical tokens suitable for model input.
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=2).to(device)

    def preprocess(data):
        return tokenizer(data["sentence1"], data["sentence2"], truncation=True, padding='max_length',
                         max_length=128)

    # prepare the dataset for training the model
    # The column "label"  indicating whether two sentences are paraphrases or not.
    # Renaming it to "labels" for Torch usage, and setting the format of the dataset to a Torch-compatible format
    # 'input_ids' and 'attention_mask' are the tokenized input representations of the sentences,
    # 'labels' contains the target labels for paraphrase detection.
    encoded_dataset = dataset.map(preprocess, batched=True, num_proc=2)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Define DataLoader for Training and Validation
    train_loader = DataLoader(encoded_dataset['train'], batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(encoded_dataset['validation'], batch_size=8, num_workers=2)

    # Train and evaluate model
    train_and_evaluate_q1(model=model, train_loader=train_loader, val_loader=val_loader)


# LoRA Fine-tune
def q2():
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=2).to(device)

    def preprocess(data):
        return tokenizer(data["sentence1"], data["sentence2"], truncation=True, padding='max_length', max_length=128)

    # prepare the dataset for training the model
    # The column "label"  indicating whether two sentences are paraphrases or not.
    # Renaming it to "labels" for Torch usage, and setting the format of the dataset to a Torch-compatible format
    # 'input_ids' and 'attention_mask' are the tokenized input representations of the sentences,
    # 'labels' contains the target labels for paraphrase detection.
    dataset = load_dataset('glue', 'mrpc')  # Example dataset for paraphrase detection
    encoded_dataset = dataset.map(preprocess, batched=True, num_proc=2)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Define DataLoader for Training and Validation
    train_loader = DataLoader(encoded_dataset['train'], batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(encoded_dataset['validation'], batch_size=8, num_workers=2)

    # LoRA Configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['query_proj', 'value_proj'],
        modules_to_save=['pooler', 'classifier']
    )

    # Train and evaluate model
    model = get_peft_model(model, lora_config)
    train_and_evaluate_q2(model=model, train_loader=train_loader, val_loader=val_loader)


# Bigger models
def q3():
    # Load the tokenizer and the first model
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
    model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-large', num_labels=2).to(device)

    def preprocess(data):
        return tokenizer(data["sentence1"], data["sentence2"], truncation=True, padding='max_length', max_length=128)

    # prepare the dataset for training the model
    # The column "label"  indicating whether two sentences are paraphrases or not.
    # Renaming it to "labels" for Torch usage, and setting the format of the dataset to a Torch-compatible format
    # 'input_ids' and 'attention_mask' are the tokenized input representations of the sentences,
    # 'labels' contains the target labels for paraphrase detection.
    dataset = load_dataset('glue', 'mrpc')  # Example dataset for paraphrase detection
    encoded_dataset = dataset.map(preprocess, batched=True, num_proc=2)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Define DataLoader for Training and Validation
    train_loader = DataLoader(encoded_dataset['train'], batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(encoded_dataset['validation'], batch_size=8, num_workers=2)

    # LoRA Configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['query_proj', 'value_proj'],
        modules_to_save=['pooler', 'classifier']
    )
    model = get_peft_model(model, lora_config)

    # Train and evaluate first model
    train_and_evaluate_q3a(model=model, train_loader=train_loader, val_loader=val_loader)

    # Same for second model
    model = AutoModelForSequenceClassification.from_pretrained('google/gemma-2b', num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')

    def preprocess(data):
        return tokenizer(data["sentence1"], data["sentence2"], truncation=True, padding='max_length', max_length=32)

    dataset = load_dataset('glue', 'mrpc')  # Example dataset for paraphrase detection
    encoded_dataset = dataset.map(preprocess, batched=True, num_proc=1)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_dataset = encoded_dataset['train']
    val_dataset = encoded_dataset['validation']
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj', 'v_proj'],
        modules_to_save=['pooler', 'classifier']
    )
    model = get_peft_model(model, lora_config)

    train_and_evaluate_q3b(model=model, train_loader=train_loader, val_loader=val_loader)


if __name__ == '__main__':
    question = input("Please enter question number: ")

    if question == "1":
        print("Starting Question 1- Full Fine-Tune")
        q1()
    elif question == "2":
        print("Starting Question 2- LoRA Fine-Tune")
        q2()
    elif question == "3":
        print("Starting Question 3- Bigger Models")
        q3()
    elif question.lower() == "all":
        print("Starting Question 1- Full Fine-Tune")
        q1()
        print("\n--------------\n")
        print("Starting Question 2- LoRA Fine-Tune")
        q2()
        print("\n--------------\n")
        print("Starting Question 3- Bigger Models")
        q3()
    else:
        print("Invalid Question Number (should be \'1\', \'2\', \'3\' or \'all\')")
