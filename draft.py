import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Disable MPS backend and force CPU or CUDA
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the MRPC dataset
dataset = load_dataset('glue', 'mrpc')

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=2).to(device)


# Tokenize the dataset
def preprocess(data):
    return tokenizer(data['sentence1'], data['sentence2'], truncation=True, padding='max_length', max_length=128)


encoded_dataset = dataset.map(preprocess, batched=True, num_proc=4)  # Set num_proc for multiprocessing

# Prepare the dataset
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define DataLoader
train_loader = DataLoader(encoded_dataset['train'], batch_size=16, shuffle=True, num_workers=4)
eval_loader = DataLoader(encoded_dataset['validation'], batch_size=16, shuffle=False, num_workers=4)

# Define training arguments
num_epochs = 3
init_learning_rate_query_value = 1e-7
end_learning_rate_query_value = 10

# Generate 10 different r factors from 4 to 32 using a logarithmic scale
r_factors = [4 * (2 ** (i / 9)) for i in range(10)]

# Record learning rates, losses, and accuracy scores
learning_rates = []
training_losses = []
accuracy_scores = []

best_accuracy = 0
optimal_r_factor = None
optimal_learning_rate = None

for r_factor in r_factors:
    optimizer_grouped_parameters = [
        {"params": [param for name, param in model.named_parameters() if "query" in name or "value" in name],
         "lr": init_learning_rate_query_value},
        {"params": [param for name, param in model.named_parameters() if "query" not in name and "value" not in name],
         "lr": 1e-7},
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    # Learning rate range test for query and value projection matrices
    lr_scheduler_query_value = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (
                                                                                                               end_learning_rate_query_value / init_learning_rate_query_value) ** (
                                                                                                               step / (
                                                                                                                   len(train_loader) * num_epochs * r_factor)))

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/3")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            lr_scheduler_query_value.step()

            # Record learning rate and loss
            current_lr = optimizer_grouped_parameters[0]['lr']
            learning_rates.append(current_lr)  # Record LR for query and value projection matrices
            training_losses.append(loss.item())

        print(f"Train loss for epoch {epoch + 1}: {sum(train_losses) / len(train_losses):.4f}")

    # Evaluate model accuracy
    model.eval()
    preds = []
    true_labels = []

    for batch in tqdm(eval_loader, desc=f"Evaluating for r_factor={r_factor}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].numpy()

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

        preds.extend(predictions)
        true_labels.extend(labels)

    accuracy = accuracy_score(true_labels, preds)
    accuracy_scores.append(accuracy)
    print(f"Accuracy for r_factor={r_factor}: {accuracy:.4f}")

    # Update the best accuracy and corresponding r factor and learning rate
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        optimal_r_factor = r_factor
        optimal_learning_rate = current_lr

print(f"Optimal r factor: {optimal_r_factor}")
print(f"Optimal learning rate: {optimal_learning_rate}")

# Save model
model.save_pretrained('./deberta_model')
