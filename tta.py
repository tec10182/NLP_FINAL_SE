import argparse
import os
import numpy as np
import random
from tqdm import tqdm
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

from model import *
from dataset import *
from loss import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def obtain_label(input, netF, netB, netC, device, distance="cosine", threshold=5):
    start_test = True
    with torch.no_grad():
        input_ids = input["input_ids"].to(device)
        attention_mask = input["attention_mask"].to(device)
        labels = input["label"]

        outputs = netF(input_ids=input_ids, attention_mask=attention_mask)
        features = netB(outputs.last_hidden_state)
        logits = netC(features)

        features = features[:, -1, :]

        if start_test:
            all_fea = features.cpu()
            all_output = logits.cpu()
            all_label = labels
            start_test = False
        else:
            all_fea = torch.cat((all_fea, features.cpu()), 0)
            all_output = torch.cat((all_output, logits.cpu()), 0)
            all_label = torch.cat((all_label, labels), 0)

    all_output = F.softmax(all_output, dim=1)
    entropy = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
    unknown_weight = 1 - entropy / np.log(all_output.size(1))
    _, predict = torch.max(all_output, 1)

    acc1 = (predict == all_label).float().mean().item()

    if distance == "cosine":
        all_fea = torch.cat([all_fea, torch.ones(all_fea.size(0), 1)], dim=1)
        all_fea = F.normalize(all_fea, p=2, dim=1)

    all_fea = all_fea.numpy()
    K = all_output.size(1)
    aff = all_output.numpy()

    for _ in range(2):
        initc = aff.T.dot(all_fea) / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > threshold)[0]

        dd = cdist(all_fea, initc[labelset], distance)
        pred_label = labelset[dd.argmin(axis=1)]
        predict = pred_label
        aff = np.eye(K)[predict]

    acc2 = np.mean(predict == all_label.numpy())
    # print(f"Accuracy = {acc1 * 100:.2f}% -> {acc2 * 100:.2f}%\n")

    return predict.astype("int")


set_seed(42)


parser = argparse.ArgumentParser(description="SE TTA")
parser.add_argument("--epoch", type=int, default=50, help="Number of training epochs")
parser.add_argument(
    "--batch_size", type=int, default=128, help="Batch size for training"
)
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer"
)
parser.add_argument("--source", required=True, help="source dataset path")
parser.add_argument("--target", required=True, help="target dataset path")
parser.add_argument("--freeze", help="freeze model")
parser.add_argument(
    "--distance", type=str, default="cosine", choices=["euclidean", "cosine"]
)

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
netC = Classifier(hidden_size=mid_hidden_size, output_size=2).to(device)

netF.load_state_dict(
    torch.load(
        os.path.join("outputs/base", args.source, "netF.pth"),
        map_location=device,
    )
)
netB.load_state_dict(
    torch.load(
        os.path.join("outputs/base", args.source, "netB.pth"),
        map_location=device,
    )
)
netC.load_state_dict(
    torch.load(
        os.path.join("outputs/base", args.source, "netC.pth"),
        map_location=device,
    )
)

# freeze model
if args.freeze == "f":
    for param in netF.parameters():
        param.requires_grad = False
elif args.freeze == "b":
    for param in netB.parameters():
        param.requires_grad = False
elif args.freeze == "c":
    for param in netC.parameters():
        param.requires_grad = False

prefix = "datasets/amazon"

# test_dataset = AmazonDataset(os.path.join(prefix, args.target, "train.txt"), tokenizer)
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

if args.target == "sst":
    test_dataset = StsDataset(file_path="datasets/sst/dev.tsv", tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
else:
    test_dataset = AmazonDataset(
        os.path.join(prefix, args.target, "train.txt"), tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

params = (
    list(netF.parameters()) + list(netB.parameters()) + list(netC.parameters())
)  # freeze model
optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
criterion = Entropy

acc = evaluate(netF, netB, netC, test_loader, device)
print(acc)

total_loss = 0.0
for epoch in range(args.epoch):
    netF.train()
    netB.train()
    netC.train()

    total_loss = 0
    loop = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epoch}")
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = netF(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state

        features = netB(embeddings)
        logits = netC(features)

        netF.eval()
        netB.eval()
        netC.eval()
        mem_label = obtain_label(batch, netF, netB, netC, device)
        mem_label = torch.from_numpy(mem_label).cuda()
        netF.train()
        netB.train()
        netC.train()

        # classifier_loss = nn.CrossEntropyLoss()(logits, mem_label)
        # classifier_loss *= 0.3  # 0.3
        classifier_loss = 0

        softmax_out = F.softmax(logits, dim=1)
        entropy_loss = torch.mean(Entropy(softmax_out))

        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
        entropy_loss -= gentropy_loss

        im_loss = entropy_loss * 1
        classifier_loss += im_loss

        classifier_loss.backward()
        optimizer.step()

        total_loss += classifier_loss.item()
        loop.set_postfix(loss=classifier_loss.item())

    total_loss /= len(test_loader)
    acc = evaluate(netF, netB, netC, test_loader, device)

    print(
        f"Epoch [{epoch+1}/{args.epoch}] - Loss: {total_loss:.4f}, Accuracy: {acc:.4f}"
    )


save_dir = os.path.join("outputs/tta", args.target)
os.makedirs(save_dir, exist_ok=True)

torch.save(netF.state_dict(), os.path.join(save_dir, "netF.pth"))
torch.save(netB.state_dict(), os.path.join(save_dir, "netB.pth"))
torch.save(netC.state_dict(), os.path.join(save_dir, "netC.pth"))
