
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
from transformers import DataCollatorForLanguageModeling
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 32
    MAX_LEN = 512
    NUM_EPOCHS = 15
    LEARNING_RATE = 5e-5
    WARMUP_STEPS = 1000
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")


    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=MAX_LEN,
        n_ctx=MAX_LEN,
        n_embd=384,
        n_layer=6,
        n_head=6
    )

    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load("best_model.pth"))
    model = model.to(device)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN, padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    train_loader = DataLoader(
        tokenized_datasets["train"], 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=data_collator,
        num_workers = 8
    )
    val_loader = DataLoader(
        tokenized_datasets["validation"], 
        batch_size=BATCH_SIZE, 
        collate_fn=data_collator
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=WARMUP_STEPS, 
        num_training_steps=len(train_loader) * NUM_EPOCHS
    )

    def train(model, train_loader, optimizer, scheduler, epoch):
        model.train()
        total_loss = 0.
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]")
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        return total_loss / len(train_loader)

    def evaluate(model, val_loader):
        model.eval()
        total_loss = 0.
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="[VALID]")
            for batch in progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                progress_bar.set_postfix({'loss': loss.item()})
        return total_loss / len(val_loader)

    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, optimizer, scheduler, epoch)
        val_loss = evaluate(model, val_loader)
        print(f'Epoch: {epoch+1}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    print('Training finished')


    def generate_text(model, prompt, max_length=100):
        model.eval()
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95)
        
        return tokenizer.decode(output[0], skip_special_tokens=True)

    prompt = "Once upon a time"
    generated_text = generate_text(model, prompt)
    print(f"Generated text:\n{generated_text}")