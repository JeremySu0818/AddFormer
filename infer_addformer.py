import argparse
import os
import torch
import json
from dataclasses import dataclass
from transformers import LlamaForCausalLM, GenerationConfig


@dataclass
class MathTokenizer:
    def __init__(self, model_max_length=64):
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
        self.model_max_length = model_max_length

    @classmethod
    def from_pretrained(cls, path):
        config_path = os.path.join(path, "tokenizer_config.json")
        vocab_path = os.path.join(path, "vocab.json")

        tokenizer = cls()
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                tokenizer.model_max_length = config.get("model_max_length", 64)

        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
                tokenizer.token_to_id = vocab
                tokenizer.id_to_token = {int(v): k for k, v in vocab.items()}

        return tokenizer

    def __call__(self, texts, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]

        input_ids_list = []
        attention_mask_list = []

        for text in texts:
            ids = [self.token_to_id.get(c, self.unk_token_id) for c in text]

            input_ids_list.append(ids)
            attention_mask_list.append([1] * len(ids))

        if return_tensors == "pt":
            max_len = max(len(x) for x in input_ids_list)
            padded_ids = []
            padded_mask = []

            for ids, mask in zip(input_ids_list, attention_mask_list):
                pad_len = max_len - len(ids)
                if self.padding_side == "left":
                    ids = [self.pad_token_id] * pad_len + ids
                    mask = [0] * pad_len + mask
                else:
                    ids = ids + [self.pad_token_id] * pad_len
                    mask = mask + [0] * pad_len
                padded_ids.append(ids)
                padded_mask.append(mask)

            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            }

        return {"input_ids": input_ids_list, "attention_mask": attention_mask_list}

    def decode(self, token_ids, skip_special_tokens=False):
        res = ""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        for idx in token_ids:
            char = self.id_to_token.get(idx, "<unk>")
            if skip_special_tokens and char in ["<pad>", "<s>", "</s>"]:
                continue
            res += char
        return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="addformer")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")

    try:
        tokenizer = MathTokenizer.from_pretrained(args.model_path)
        model = LlamaForCausalLM.from_pretrained(args.model_path)
        model.to(args.device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model loaded. Enter equations (e.g., '1+1='). Type 'exit' to quit.")

    while True:
        try:
            text = input("Input: ").strip()
            if text.lower() in ["exit", "quit"]:
                break
            if not text:
                continue

            if "=" not in text:
                text += "="

            inputs = tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(args.device)
            attention_mask = inputs["attention_mask"].to(args.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    repetition_penalty=1.1,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Output: {generated_text}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during inference: {e}")


if __name__ == "__main__":
    main()
