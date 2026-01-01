import argparse
import random
from pathlib import Path


def iter_range_with_digits(d):
    start = 10 ** (d - 1)
    end = 10**d - 1
    return start, end


def enumerate_all_pairs(max_digits):
    groups = {}
    for da in range(1, max_digits + 1):
        for db in range(1, max_digits + 1):
            a_start, a_end = iter_range_with_digits(da)
            b_start, b_end = iter_range_with_digits(db)
            pairs = [
                (a, b)
                for a in range(a_start, a_end + 1)
                for b in range(b_start, b_end + 1)
            ]
            groups[(da, db)] = pairs
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--max_digits", type=int, default=3)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    groups = enumerate_all_pairs(args.max_digits)
    groups_list = list(groups.keys())

    train_f = open(out / "train.txt", "w", encoding="utf-8")

    train_pairs = []
    for key in groups_list:
        for a, b in groups[key]:
            train_f.write(f"{a}+{b}={a+b}\n")
            train_pairs.append((a, b))

    train_f.close()

    val_pairs = train_pairs

    with open(out / "val.txt", "w", encoding="utf-8") as f:
        for a, b in val_pairs:
            f.write(f"{a}+{b}={a+b}\n")

    print(f" - train.txt = {len(train_pairs)} 行")
    print(f" - val.txt   = {len(val_pairs)} 行")


if __name__ == "__main__":
    main()
