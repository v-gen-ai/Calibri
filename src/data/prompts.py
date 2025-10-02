from torch.utils.data import Dataset, DataLoader

def get_lines(path, limit=-1):
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    prompts = [l for l in lines if len(l) > 0]
    
    if limit is not None and limit > 0:
        prompts = prompts[:limit]
    return prompts


class PromptDataset(Dataset):
    def __init__(self, path, limit=-1):
        self.prompts = get_lines(path, limit)

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


class CutDataLoader:
    """Простая бесконечная обёртка над любым DataLoader."""
    def __init__(self, base_loader, max_cnt = 10):
        self.base_loader = base_loader
        self.max_cnt = max_cnt

    def __iter__(self):
        cnt = 0
        for batch in self.base_loader:
            cnt += 1
            if cnt > self.max_cnt:
                break
            yield batch

    def __len__(self):
        return self.max_cnt

def make_loader(path, batch_size, num_workers, shuffle, drop_last, limit=-1, infinite=False, cut_cnt=None):
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
    elif cut_cnt is not None:
        return CutDataLoader(base, cut_cnt)
    return base
