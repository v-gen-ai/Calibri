from typing import List
import json
from torch.utils.data import Dataset, DataLoader

def get_lines(path: str, limit: int = -1) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lines: List[str] = []

        if isinstance(data, list):
            if all(isinstance(x, str) for x in data):
                lines = data
            elif all(isinstance(x, dict) and "prompt" in x for x in data):
                lines = [str(x["prompt"]) for x in data if isinstance(x.get("prompt"), (str, int, float))]
            else:
                lines = [str(x) for x in data]
        elif isinstance(data, dict):
            if "prompts" in data and isinstance(data["prompts"], list):
                cand = data["prompts"]
                if all(isinstance(x, str) for x in cand):
                    lines = cand
                elif all(isinstance(x, dict) and "prompt" in x for x in cand):
                    lines = [str(x["prompt"]) for x in cand if isinstance(x.get("prompt"), (str, int, float))]
                else:
                    lines = [str(x) for x in cand]
            elif "lines" in data and isinstance(data["lines"], list):
                lines = [str(x) for x in data["lines"]]
            else:
                lines = [str(v) for v in data.values() if isinstance(v, (str, int, float))]
        else:
            lines = [str(data)]

    except json.JSONDecodeError:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f]

    prompts = [l.strip() for l in lines if isinstance(l, str) and l.strip()]

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
    def __init__(self, base_loader):
        self.base_loader = base_loader

    def __iter__(self):
        while True:
            for batch in self.base_loader:
                yield batch


class CutDataLoader:
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
