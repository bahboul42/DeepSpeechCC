from deepspeech import CommonVoiceDataset, DeepSpeechCC
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import wandb

from tqdm import tqdm

WANDB_FLAG = True

# Hyperparameters
lr = 3e-4
batch_size = 32
n_epochs = 3
model_name = 'deepspeechCC-ver-0.0'

batch_test_size = 3

eval_every = 50
save_every = 300

directory = "./models"
checkpoint = f"{directory}/{model_name}.pt"

if WANDB_FLAG:
    wandb.init(
        project="deepspeech-CC",
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": lr,
            "n_epochs": n_epochs
        }
    )

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'mps'

# paths
ROOT_DIR = '/Users/rayaneeloudrhiri/dataset/cv-corpus-17.0-2024-03-15/en/'

if __name__ == "__main__":
    # Load data
    train_dataset = CommonVoiceDataset(root_dir=ROOT_DIR, mode='train', sample=0.05)
    val_dataset = CommonVoiceDataset(root_dir=ROOT_DIR, mode='val', sample=0.05)
    test_dataset = CommonVoiceDataset(root_dir=ROOT_DIR, mode='test', sample=0.05)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_test_size, shuffle=True)

    for melspecs, sentences in test_loader:
        test_melspecs, test_sentences = melspecs, sentences
        break
    
    # Get model
    model = DeepSpeechCC()
    model = model.to(device)

    print("Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = None

    loss_fn = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    vocab_size = model.vocab_size
    
    iter = 0
    for epoch in tqdm(range(n_epochs), desc='Epochs', position=0):
        model.train()
        with tqdm(range(len(train_loader)), desc='Batches', position=1) as pbar:
            for melspecs, sentences in train_loader:
                tokenized_sentences = model.label_parser(sentences)
                melspecs, tokenized_sentences = melspecs.to(device), tokenized_sentences.to(device)
                optimizer.zero_grad()

                # forward pass
                logits = model(melspecs, tokenized_sentences)
                # reshape logits to only keep the relevant part i.e. transcription prediction
                encoded_melspecs_size = int(melspecs.shape[2]/2)
                relevant_logits = logits[:, encoded_melspecs_size-1:-1, :]
                # compute loss between logits and tokenized_sentences
                loss = loss_fn(relevant_logits.reshape(-1, vocab_size), tokenized_sentences.reshape(-1))

                # backward pass
                loss.backward()
                optimizer.step()
                iter += 1
                if iter % save_every == 0:
                    torch.save(model.state_dict(), f'{checkpoint}_{iter}')

                if iter % eval_every == 0:
                    out = {}
                    model.eval()
                    with torch.no_grad():
                        for split, loader in [('train', train_loader), ('validation', val_loader)]:
                            k=0
                            with tqdm(range(len(loader)), desc=f'Evaluation in {split} dataset...', position=2) as pbar2:
                                losses = torch.zeros(len(loader))
                                for melspecs, sentences in loader:
                                    tokenized_sentences = model.label_parser(sentences)
                                    melspecs, tokenized_sentences = melspecs.to(device), tokenized_sentences.to(device)
                                    logits = model(melspecs, tokenized_sentences)
                                    encoded_melspecs_size = int(melspecs.shape[2]/2)
                                    relevant_logits = logits[:, encoded_melspecs_size-1:-1, :]
                                    losses[k] = loss_fn(relevant_logits.reshape(-1, vocab_size), tokenized_sentences.reshape(-1)).item()
                                    k+=1
                                    pbar2.update(1)
                                out[split] = losses.mean()
                                pbar2.close()

                    model.train()
                    print(f'-----------------------------------------------------------------------------------', flush=True)
                    print(f"step {iter}: train loss {out['train']:.4f}, validation loss {out['validation']:.4f}", flush=True)
                    print(f'-----------------------------------------------------------------------------------', flush=True)

                    # try model
                    model.eval()
                    with torch.no_grad():
                        next_token = ["" for _ in range(test_melspecs.shape[0])]
                        encoded_sentence_test = model.label_parser(test_sentences)
                        max_len = encoded_sentence_test.shape[1]
                        logits = model(test_melspecs.to(device))
                        relevant_logits = logits[:, -1:, :] # only keep the last token
                        preds = torch.argmax(relevant_logits, dim=-1)
                        next_token = [nt + model.tokenizer.decode(pred) for nt, pred in zip(next_token, preds)]
                        print_sentence = [s for s in test_sentences]
                        print(f'Targets: {print_sentence}', flush=True)
                        for i in range(1, max_len):
                            next_token_encoded = model.label_parser(next_token)
                            logits = model(test_melspecs.to(device), next_token_encoded.to(device))
                            encoded_melspecs_size = int(test_melspecs.shape[2]/2)
                            relevant_logits = logits[:, -1:, :] # only keep the last token
                            preds = torch.argmax(relevant_logits, dim=-1)

                            # decode predictions
                            next_token = [nt + model.tokenizer.decode(pred) for nt, pred in zip(next_token, preds)]
                        
                        print(f'Predictions: {next_token}', flush=True)
                    model.train()
                    print_sentence_data = [[s] for s in print_sentence]
                    next_token_data = [[nt] for nt in next_token]
                    wandb.log({
                        "step": iter,
                        "training_loss": out['train'],
                        "validation_loss": out['validation'],
                        "sentences": wandb.Table(columns=["sentences"], data=print_sentence_data),
                        "predictions": wandb.Table(columns=["predictions"], data=next_token_data)
                    })

                pbar.update(1)
