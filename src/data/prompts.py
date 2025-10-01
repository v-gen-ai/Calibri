from torch.utils.data import Dataset, DataLoader

class PromptDataset(Dataset):
    def __init__(self, path, limit=-1):
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
        self.prompts = [l for l in lines if len(l) > 0]
        if limit is not None and limit > 0:
            self.prompts = self.prompts[:limit]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

def collate_prompts(batch):
    # batch: list[str]
    return batch  # список промптов «как есть»

def make_loader(path, batch_size, num_workers, shuffle, drop_last, limit=-1):
    ds = PromptDataset(path, limit=limit)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=drop_last, collate_fn=collate_prompts)
