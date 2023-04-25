import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("ag_news", split="train")

# Preprocess the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Initialize the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
tokenized_texts = dataset.map(tokenize_function, batched=True)

# Load the tokenized dataset
train_dataset = tokenized_texts.remove_columns(["text"]).select(range(2000))

# Define the training loop
def train_loop(dataloader, model, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone().detach()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Prepare the data
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Set up the model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Train the model
num_epochs = 3

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train_loop(train_dataloader, model, optimizer, device)
    print(f"Training loss: {train_loss:.4f}")

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
