import argparse
import os
import re
import torch
from dataclasses import dataclass
from typing import List, Optional
from transformers import LlamaForCausalLM


@dataclass
class MathTokenizer:
    def __init__(self):
        self.chars = [
            "<pad>",
            "<s>",
            "</s>",
            "<unk>",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "+",
            "-",
            "*",
            "/",
            "=",
            ".",
            "(",
            ")",
            "^",
            "%",
            " ",
        ]
        self.token_to_id = {c: i for i, c in enumerate(self.chars)}
        self.id_to_token = {i: c for i, c in enumerate(self.chars)}
        self.pad_token_id = self.token_to_id["<pad>"]
        self.eos_token_id = self.token_to_id["</s>"]
        self.bos_token_id = self.token_to_id["<s>"]
        self.unk_token_id = self.token_to_id["<unk>"]
        self.padding_side = "left"

    def __call__(self, texts, return_tensors=None, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        input_ids_list = [
            [self.token_to_id.get(c, self.unk_token_id) for c in t] for t in texts
        ]

        if return_tensors == "pt":
            return {"input_ids": torch.tensor(input_ids_list, dtype=torch.long)}
        return {"input_ids": input_ids_list}

    def decode(self, token_ids, skip_special_tokens=False):
        res = ""
        for i in token_ids:
            idx = i.item() if hasattr(i, "item") else i
            char = self.id_to_token.get(idx, "<unk>")
            if skip_special_tokens and char in ["<pad>", "<s>", "</s>"]:
                continue
            res += char
        return res

    def __len__(self):
        return len(self.chars)


class ConsoleUI:
    @staticmethod
    def info(msg: str):
        print(f"[*] {msg}")

    @staticmethod
    def warn(msg: str):
        print(f"[!] {msg}")

    @staticmethod
    def error(msg: str):
        print(f"[x] {msg}")

    @staticmethod
    def header(ckpt: str, device: str, tokens: int):
        print("-" * 50)
        print(f" AddFormer Interactive Mode (Jeremy's Edition)")
        print(f" Device: {device} | Max New Tokens: {tokens}")
        print(f" Ckpt: {ckpt}")
        print("-" * 50)

    @staticmethod
    def help():
        print("\n:help - 顯示幫助 | :config - 顯示配置 | :max N - 設長度 | :q - 退出\n")


def get_sorted_checkpoints(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    entries = [d for d in os.listdir(base_dir) if re.match(r"checkpoint-(\d+)$", d)]
    return sorted(entries, key=lambda x: int(x.split("-")[1]), reverse=True)


def select_checkpoint_interactive(root: str, ckpts: List[str]) -> Optional[str]:
    if not ckpts:
        return None
    try:
        import msvcrt

        idx = 0
        while True:
            os.system("cls")
            print(
                f"Select Checkpoint (Root: {root})\nUse Arrows, Enter to confirm, Q to quit.\n"
            )
            for i, name in enumerate(ckpts):
                print(f"{'->' if i == idx else '    '} {name}")
            ch = msvcrt.getch()
            if ch in (b"\xe0", b"\x00"):
                code = msvcrt.getch()
                if code == b"H":
                    idx = (idx - 1) % len(ckpts)
                elif code == b"P":
                    idx = (idx + 1) % len(ckpts)
            elif ch in (b"\r", b"\n"):
                return os.path.join(root, ckpts[idx])
            elif ch.lower() == b"q":
                return None
    except ImportError:
        print("Available Checkpoints:")
        for i, c in enumerate(ckpts):
            print(f"{i}: {c}")
        choice = input("Select index: ")
        return os.path.join(root, ckpts[int(choice)]) if choice.isdigit() else None


def generate_text(
    model, tokenizer, prompt: str, device: torch.device, max_tokens: int
) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="best_models")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        ckpts = get_sorted_checkpoints("addformer_ckpt")
        ckpt_path = select_checkpoint_interactive("addformer_ckpt", ckpts) or args.ckpt

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )

    ConsoleUI.info(f"Loading Model from {ckpt_path}...")

    tokenizer = MathTokenizer()

    try:
        model = LlamaForCausalLM.from_pretrained(ckpt_path)
        model.to(device)
        model.eval()
    except Exception as e:
        ConsoleUI.error(f"Failed to load model: {e}")
        return

    max_new_tokens = args.max_new_tokens
    ConsoleUI.header(ckpt_path, str(device), max_new_tokens)

    while True:
        try:
            user_in = input(">>> ").strip()
        except EOFError:
            break

        if not user_in:
            continue
        if user_in.startswith(":"):
            cmd = user_in[1:].lower()
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "help":
                ConsoleUI.help()
            elif cmd.startswith("max "):
                try:
                    max_new_tokens = int(cmd.split()[1])
                except:
                    ConsoleUI.error("Invalid number")
            continue

        prompt = user_in if "=" in user_in else user_in + "="

        result = generate_text(model, tokenizer, prompt, device, max_new_tokens)
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
