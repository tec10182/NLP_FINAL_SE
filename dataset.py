from torch.utils.data import Dataset

class AmazonDataset(Dataset):
    def __init__(self, file_path, tokenizer=None, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '|||' not in line:
                    continue 
                text, label = line.strip().rsplit('|||', 1)
                self.data.append((text.strip(), int(label.strip())))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            # squeeze로 [1, 512] -> [512] 등 차원 정리
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'label': label
            }
        else:
            return text, label