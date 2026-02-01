from pathlib import Path


def main():
    out_dir = Path("data_add")
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for a in range(10):
        for b in range(10):
            pairs.append(f"{a}+{b}={a+b}")

    with open(out_dir / "train.txt", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(p + "\n")

    with open(out_dir / "val.txt", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(p + "\n")

    print(f"Add generated: {len(pairs)} lines")


if __name__ == "__main__":
    main()
