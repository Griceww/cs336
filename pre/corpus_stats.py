import re
from dataclasses import dataclass, field
import argparse
import os
import sys

def load_text(path: str, encoding: str = "utf-8") -> str :
    with open(path, "r", encoding=encoding) as f:
        return f.read()

def tokenize(text: str, lang: str | None = None) -> list[str] :
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return tokens

@dataclass
class CorpusStats :
    token_freq: dict[str, int] = field(default_factory=dict)  # 词 -> 频次
    total_tokens: int          = 0                            # token 总数
    vocab_size: int            = 0                            # 不同 token 数

def compute_stats(tokens: list[str]) -> CorpusStats :
    stats = CorpusStats()
    for w in tokens :
        stats.token_freq[w] = stats.token_freq.get(w, 0) + 1
    stats.total_tokens = len(tokens)
    stats.vocab_size = len(stats.token_freq)
    return stats

def format_stats(stats: CorpusStats, topk: int = 20) -> str :
    items = list(stats.token_freq.items())
    items = sorted(items, key = lambda pair : (-pair[1], pair[0]))
    top_items = items[:topk]
    lines : list[str] = []
    lines.append("=== Corpus Stats ===")
    lines.append(f"Total tokens : {stats.total_tokens}")
    lines.append(f"Vocab size   : {stats.vocab_size}")
    lines.append("")
    lines.append(f"{'Rank' : >4}  {'Token' : <15}  {'Freq' : >8}")
    rank = 0
    for token, count in top_items :
        rank = rank + 1
        lines.append(f"{rank : >4}  {token : <15}  {count : >8}")
    return "\n".join(lines)

def positive_int(value : str) -> int :
    try :
        ivalue = int(value)
    except ValueError :
        raise argparse.ArgumentTypeError(
            f"invalid int value: {value!r}"
        )
    if ivalue <= 0 :
        raise argparse.ArgumentTypeError(
            f"--topk must be a positive integer, got {value}"
        )
    return ivalue

def main() :
    parser = argparse.ArgumentParser()

    # --input（必选）
    parser.add_argument(
        "--input",
        required=True,
        help="Input text file path"
    )

    # --topk（可选，默认 20）
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Top-K most frequent tokens (default: 20)"
    )

    args = parser.parse_args()

    input_path = args.input
    topk = args.topk
    
    if not os.path.isfile(input_path) :
        print(f"Error: input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Input file: {input_path}")
    print(f"Top-K: {topk}")

    stats = compute_stats(tokenize(load_text(input_path)))
    print(format_stats(stats, topk))


if __name__ == "__main__":
    main()