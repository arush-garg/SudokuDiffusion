import json


def load_jsonl(path, limit=None):
    records = []
    f = open(path)
    for i, line in enumerate(f):
        if limit is not None and i >= limit:
            break
        line = line.strip()
        if line:
            records.append(json.loads(line))
    f.close()
    return records


class SudokuSample:
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels


class SudokuDataset:
    def __init__(self, path):
        self.records = load_jsonl(path)
        if not self.records:
            raise ValueError(f"No records found in {path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]
