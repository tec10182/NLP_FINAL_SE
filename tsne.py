import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader
from model import BiGRU  # 사용자 정의 모델
from dataset import AmazonDataset  # 사용자 정의 데이터셋
import seaborn as sns  # 선택사항, 색상 팔레트 사용

device = "cuda" if torch.cuda.is_available() else "cpu"

# 경로 설정
source = "book"  # 예시 source domain
save_dir = f"workspace/NLP_FINAL/outputs/tta/book"
data_path = f"workspace/NLP_FINAL/datasets/amazon/book/train.txt"

# 모델 정의
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
netF = RobertaModel.from_pretrained("roberta-base")
mid_hidden_size = netF.config.hidden_size // 2
netB = BiGRU(input_size=netF.config.hidden_size, hidden_size=mid_hidden_size)

# 모델 불러오기
netF.load_state_dict(
    torch.load(os.path.join(save_dir, "netF.pth"), map_location=device)
)
netB.load_state_dict(
    torch.load(os.path.join(save_dir, "netB.pth"), map_location=device)
)

netF.to(device).eval()
netB.to(device).eval()

# 데이터셋 로드
dataset = AmazonDataset(data_path, tokenizer)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# 특징 추출
features = []
labels = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].cpu().numpy()

        outputs = netF(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state  # (B, T, H)
        pooled = netB(embedding)  # (B, H)

        pooled = pooled[:, -1, :]

        features.append(pooled.cpu().numpy())
        labels.extend(label)

features = torch.cat([torch.tensor(f) for f in features], dim=0).numpy()

# TSNE 수행
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# 시각화
plt.figure(figsize=(10, 7))

# 고정된 색상 지정
colors = {0: "red", 1: "blue"}  # 또는 sns.color_palette("hsv", 2) 사용 가능

for label in set(labels):
    idxs = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(
        features_2d[idxs, 0],
        features_2d[idxs, 1],
        label=f"Class {label}",
        alpha=0.6,
        color=colors[label],
        edgecolors="k",
        s=40,
    )

plt.legend()
plt.title(f"t-SNE Visualization for {source} domain")
plt.savefig(f"tsne_{source}.png")
plt.show()


plt.legend()
plt.title(f"t-SNE Visualization for {source} domain")
plt.savefig(f"tsne_{source}.png")
plt.show()
