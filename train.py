import argparse
import os
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

from model import *
from dataset import *


def evaluate(model_F, model_B, model_C, dataloader, device):
    model_F.eval()
    model_B.eval()
    model_C.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model_F(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

            features = model_B(embeddings)
            logits = model_C(features)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)

parser = argparse.ArgumentParser(description="SE TTA")
parser.add_argument("--epoch", type=int, default=10, help="Number of training epochs")
parser.add_argument(
    "--batch_size", type=int, default=16, help="Batch size for training"
)
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer"
)
parser.add_argument("--source", required=True, help="source dataset")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
netF = RobertaModel.from_pretrained("roberta-base").to(device)

# tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
# netF = BertModel.from_pretrained("prajjwal1/bert-tiny").to(device)

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# netF = BertModel.from_pretrained("bert-base-uncased").to(device)


mid_hidden_size = netF.config.hidden_size // 2  # 256
netB = BiGRU(input_size=netF.config.hidden_size, hidden_size=mid_hidden_size).to(device)
netC = Classifier(hidden_size=mid_hidden_size, output_size=2).to(device)


prefix = "workspace/NLP_FINAL/datasets/amazon"

domains = ["book", "dvd", "electronics", "kitchen"]
loaders = {}

if args.source == "sst":
    dataset = StsDataset(
        file_path="workspace/NLP_FINAL/datasets/sst/train.tsv", tokenizer=tokenizer
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
else:
    for domain in domains:
        dataset = AmazonDataset(
            os.path.join(os.path.join(prefix, domain), "train.txt"), tokenizer
        )
        loaders[domain] = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    train_loader = loaders[args.source]
    val_domains = [loaders[domain] for domain in domains if domain != args.source]


params = list(netF.parameters()) + list(netB.parameters()) + list(netC.parameters())
optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()


save_dir = os.path.join("workspace/NLP_FINAL/outputs/base", args.source)
os.makedirs(save_dir, exist_ok=True)


total_loss = 0.0
for epoch in range(args.epoch):
    netF.train()
    netB.train()
    netC.train()

    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}")
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = netF(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state

        features = netB(embeddings)
        logits = netC(features)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    total_loss /= len(train_loader)
    # for val in val_domains:
    #     acc = evaluate(netF, netB, netC, val, device)
    #     print(f"Accuracy: {acc:.4f}")

    print(f"Epoch [{epoch+1}/{args.epoch}] - Loss: {total_loss:.4f}")
    torch.save(netF.state_dict(), os.path.join(save_dir, "netF.pth"))
    torch.save(netB.state_dict(), os.path.join(save_dir, "netB.pth"))
    torch.save(netC.state_dict(), os.path.join(save_dir, "netC.pth"))
