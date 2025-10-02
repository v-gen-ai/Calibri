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

class InfiniteDataLoader:
    """Простая бесконечная обёртка над любым DataLoader."""
    def __init__(self, base_loader):
        self.base_loader = base_loader

    def __iter__(self):
        while True:
            for batch in self.base_loader:
                yield batch

def make_loader(path, batch_size, num_workers, shuffle, drop_last, limit=-1, infinite=False):
    ds = PromptDataset(path, limit=limit)
    base = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    if infinite:
        return InfiniteDataLoader(base)
    return base
