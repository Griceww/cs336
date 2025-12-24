import torch

INPUT_IDS_KEY: str = "input_ids"
LABELS_KEY: str = "labels"

class TextDataset :
    _tokens : torch.Tensor
    _seq_len : int
    _stride : int
    _return_labels: bool
    _label_shift: int
    _num_samples : int

    @property
    def return_labels(self) -> bool :
        return self._return_labels

    def __init__(
        self,
        tokens,
        seq_len : int,
        stride : int | None = None,
        return_labels : bool = True,
        label_shift : int = 1
    ) -> None :
        self._tokens = torch.as_tensor(tokens, dtype = torch.long)
        self._tokens = self._tokens.flatten()
        if seq_len <= 0 :
            raise ValueError("seq_len")
        self._seq_len = seq_len

        if stride == None :
            stride = seq_len
        if stride <= 0 :
            raise ValueError("stride") 
        self._stride = stride
    
        self._return_labels = return_labels
        self._label_shift = label_shift

        max_start : int
        if return_labels == True :
            if label_shift <= 0 :
                raise ValueError("label_shift")
            max_start = (len(self._tokens) - label_shift - seq_len)
        else :
            max_start = len(self._tokens) - seq_len
            
        if max_start < 0 :
            self._num_samples = 0
        else :
            self._num_samples = max_start // stride + 1

    def __len__(self) -> int :
        return self._num_samples

    def __getitem__(self, index : int) -> dict[str, torch.Tensor] :
        if index < 0 or index >= self._num_samples :
            raise IndexError("index out of range")
        sample : dict[str, torch.Tensor] = {}
        start = index * self._stride
        end = start + self._seq_len
        sample[INPUT_IDS_KEY] = self._tokens[start : end]
        if self._return_labels == True :
            sample[LABELS_KEY] = self._tokens[start + self._label_shift : 
                                              end + self._label_shift]
        return sample

def main() :
    sample = TextDataset([1, 3, 0, 4, 5], 2)
    sample_len = len(sample)
    print(type(sample_len))
    ret = sample[sample_len - 1]
    print(ret[LABELS_KEY])
    # ret = sample[sample_len]

if __name__ == "__main__" :
    main()