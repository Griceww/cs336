from multiprocessing import Value


class Vocab :
    _token_to_id : dict[str, int]
    _id_to_token : list[str]
    _unk_token : str = "<unk>"
    _unk_id : int
    _pad_token : str
    _pad_id : int | None
    _bos_token : str | None
    _bos_id : int | None
    _eos_token : str | None
    _eos_id : int | None

    def __init__(
        self, 
        token_to_id : dict[str, int], 
        unk_token : str = "<unk>",
        pad_token : str | None = None,
        bos_token : str | None = None,
        eos_token : str | None = None
    ) :
        self._token_to_id = token_to_id
        self._unk_token = unk_token
        self._pad_token = pad_token
        if unk_token not in token_to_id :
            raise ValueError("unk_token not in token_to_id")
        if pad_token :
            if pad_token not in token_to_id :
                raise ValueError("pad_token not in token_to_id")
            self._pad_id = token_to_id.get(pad_token)
        else :
            self._pad_id = None

        if bos_token :
            if bos_token not in token_to_id :
                raise ValueError(f"{bos_token} not in token_to_id")
            self._bos_token = bos_token
            self._bos_id = token_to_id.get(bos_token)
        else :
            self._bos_token = None
            self._bos_id = None

        if eos_token :
            if eos_token not in token_to_id :
                raise ValueError(f"{eos_token} not in token_to_id")
            self._eos_token = eos_token
            self._eos_id = token_to_id.get(eos_token)
        else :
            self._eos_token = None
            self._eos_id = None

        n = len(token_to_id)
        id_to_token = [""] * n
        for token, idx in token_to_id.items() :
            if idx < 0 or idx >= n :
                raise ValueError("id out of range")
            id_to_token[idx] = token
        self._id_to_token = id_to_token
        self._unk_id = token_to_id.get(unk_token)

    @property
    def unk_token(self) -> str :
        return self._unk_token

    @property
    def unk_id(self) -> int :
        return self._unk_id

    @property
    def pad_token(self) -> str | None :
        return self._pad_token
    
    @property
    def pad_id(self) -> int | None :
        return self._pad_id

    @property
    def bos_token(self) -> str | None :
        return self._bos_token
    
    @property
    def bos_id(self) -> int | None :
        return self._bos_id

    @property
    def eos_token(self) -> str | None :
        return self._eos_token
    
    @property
    def eos_id(self) -> int | None :
        return self._eos_id
    
    def token_to_id(self, token : str) -> int :
        return self._token_to_id.get(token, self._unk_id)
    
    def id_to_token(self, idx : int) -> str :
        if idx < 0 or idx >= len(self._id_to_token) :
            raise ValueError("id out of range")
        return self._id_to_token[idx]

    def __len__(self) -> int :
        return len(self._id_to_token)

    def encode(
        self, 
        tokens : list[str], 
        add_bos : bool = False, 
        add_eos : bool = False
    ) -> list[int] :
        idxs : list[int] = []
        if add_bos == True :
            if self._bos_id :
                idxs.append(self._bos_id)
            else :
                raise ValueError("bos_id not configured")

        for token in tokens :
            idxs.append(self.token_to_id(token))
        
        if add_eos == True :
            if self._eos_id :
                idxs.append(self._eos_id)
            else :
                raise ValueError("eos_id not configured")
        return idxs

    def decode(self, idxs : list[int]) -> list[str] :
        tokens : list[str] = []
        for idx in idxs :
            tokens.append(self.id_to_token(idx))
        return tokens

    def Summary(self, max_tokens : int = 20) -> str :
        lines = []
        lines.append("-----Vocab Summary-----")
        lines.append(f"Size : {len(self)}")
        lines.append(f"UNK : token = {self.unk_token}, id = {self.unk_id}")
        lines.append(f"PAD : token = {self.pad_token}, id = {self.pad_id}")
        lines.append(f"BOS : token = {self.bos_token}, id = {self.bos_id}")
        lines.append(f"EOS : token = {self.eos_token}, id = {self.eos_id}")
        lines.append(f"{'token':>8}{'id':>8}")
        limits = min(max_tokens, len(self))
        for id in range(limits) :
            lines.append(f"{self._id_to_token[id]:>8}{id:>8}")
        return "\n".join(lines)

def main() :
    token_to_id = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "hello": 4,
        "world": 5,
    }
    v = Vocab(
        token_to_id,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )

    tokens = ["hello", "world"]
    print(v.encode(tokens))                             # 预期: [4, 5]
    print(v.encode(tokens, add_bos=True))              # 预期: [2, 4, 5]
    print(v.encode(tokens, add_eos=True))              # 预期: [4, 5, 3]
    print(v.encode(tokens, add_bos=True, add_eos=True))# 预期: [2, 4, 5, 3]
    print(v.Summary())



if __name__ == "__main__" :
    main()