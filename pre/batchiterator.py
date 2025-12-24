from textdataset import TextDataset, LABELS_KEY, INPUT_IDS_KEY
import torch
from typing import Iterator

class BatchIterator :
    _dataset : TextDataset
    _batch_size : int
    _drop_last: bool
    _shuffle : bool
    _seed : int
    _epoch : int

    def __init__(
        self,
        dataset : TextDataset,
        batch_size : int,
        drop_last : bool = True,
        shuffle : bool = False,
        seed : int = 0
    ) -> None :
        self._dataset = dataset

        if batch_size < 1 :
            raise ValueError("batch_size")
        self._batch_size = batch_size

        self._drop_last = drop_last
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

    def __len__(self) -> int :
        if self._drop_last == True :
            return len(self._dataset) // self._batch_size
        return (len(self._dataset) + self._batch_size - 1) // self._batch_size

    def _collate(self, samples : list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor] :
        batch : dict[str, torch.Tensor] = {}
        batch[INPUT_IDS_KEY] = torch.stack([s[INPUT_IDS_KEY] for s in samples], dim = 0)
        if self._dataset.return_labels == True :
            batch[LABELS_KEY] = torch.stack([s[LABELS_KEY] for s in samples], dim = 0)
        return batch

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]] :
        indices = self._make_indices()
        for index in range(len(self)):
            samples : list[dict[str, torch.Tensor]] = []
            for batch_item in range(index * self._batch_size, 
                                    min((index + 1) * self._batch_size, len(self._dataset))) :
                samples.append(self._dataset[indices[batch_item]])
            yield self._collate(samples)

    def set_epoch(self, epoch : int) -> None :
        self._epoch = epoch

    def _make_indices(self) -> list[int] :
        n = len(self._dataset)
        if self._shuffle == False :
            return list(range(n))
        gen = torch.Generator()
        gen.manual_seed(self._seed + self._epoch)
        return torch.randperm(n, generator = gen).cpu().tolist()
            
def collect_first_tokens(it: BatchIterator):
    # 收集每个 batch 的第一个 token（用来比较顺序是否一致）
    out = []
    for b in it:
        out.extend(b[INPUT_IDS_KEY][:, 0].tolist())
    return out

def test_shuffle_reproducible_same_epoch():
    ds = TextDataset(list(range(50)), seq_len=4, stride=2, return_labels=True, label_shift=1)

    it1 = BatchIterator(ds, batch_size=4, drop_last=False, shuffle=True, seed=123)
    it2 = BatchIterator(ds, batch_size=4, drop_last=False, shuffle=True, seed=123)

    it1.set_epoch(0)
    it2.set_epoch(0)

    a = collect_first_tokens(it1)
    b = collect_first_tokens(it2)
    assert a == b, "same seed + same epoch should produce identical order"

    print("test_shuffle_reproducible_same_epoch passed.")

def test_shuffle_different_epoch_changes_order():
    ds = TextDataset(list(range(50)), seq_len=4, stride=2, return_labels=True, label_shift=1)

    it = BatchIterator(ds, batch_size=4, drop_last=False, shuffle=True, seed=123)

    it.set_epoch(0)
    a = collect_first_tokens(it)

    it.set_epoch(1)
    b = collect_first_tokens(it)

    # 大概率不同；若你担心极小概率相同，可改为比较前若干项
    assert a != b, "different epoch should change shuffle order"

    # 同一 epoch 再跑一次应一致
    it.set_epoch(1)
    c = collect_first_tokens(it)
    assert b == c, "same epoch repeated should be reproducible"

    print("test_shuffle_different_epoch_changes_order passed.")

def test_no_shuffle_is_sequential():
    ds = TextDataset(list(range(20)), seq_len=4, stride=4, return_labels=False)

    it = BatchIterator(ds, batch_size=3, drop_last=False, shuffle=False, seed=999)
    it.set_epoch(100)  # 不应影响非 shuffle

    order = collect_first_tokens(it)

    # stride=4, seq_len=4 -> 每个样本 input_ids 的第一个 token 是 0,4,8,12,16
    assert order == [0, 4, 8, 12, 16], f"expected sequential order, got {order}"

    print("test_no_shuffle_is_sequential passed.")

if __name__ == "__main__":
    test_shuffle_reproducible_same_epoch()
    test_shuffle_different_epoch_changes_order()
    test_no_shuffle_is_sequential()
    print("All shuffle tests passed.")