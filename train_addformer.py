import os
from dataclasses import dataclass
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    LlamaConfig,
    LlamaForCausalLM,
)
from transformers import logging as hf_logging
from datasets import logging as ds_logging

hf_logging.set_verbosity_error()
ds_logging.set_verbosity_error()

DATA_DIR = "data_add"
OUTPUT_DIR = "addformer_result"
MODEL_DIM = 64
N_LAYER = 8
N_HEAD = 8
MAX_LENGTH = 64
SEED = 42


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
            "Q",
            "R",
        ]
        self.token_to_id = {c: i for i, c in enumerate(self.chars)}
        self.id_to_token = {i: c for i, c in enumerate(self.chars)}
        self.pad_token = "<pad>"
        self.pad_token_id = self.token_to_id["<pad>"]
        self.eos_token = "</s>"
        self.eos_token_id = self.token_to_id["</s>"]
        self.bos_token = "<s>"
        self.bos_token_id = self.token_to_id["<s>"]
        self.unk_token = "<unk>"
        self.unk_token_id = self.token_to_id["<unk>"]
        self.padding_side = "right"
        self.model_max_length = MAX_LENGTH

    def __call__(
        self,
        texts,
        truncation=True,
        max_length=None,
        padding=False,
        add_special_tokens=False,
        return_tensors=None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        input_ids_list = []
        attention_mask_list = []
        for text in texts:
            ids = [self.token_to_id.get(c, self.unk_token_id) for c in text]
            if truncation and max_length is not None:
                ids = ids[:max_length]
            input_ids_list.append(ids)
            attention_mask_list.append([1] * len(ids))
        if padding:
            longest = max(len(ids) for ids in input_ids_list)
            if max_length and padding == "max_length":
                longest = max_length
            final_ids = []
            final_mask = []
            for ids, mask in zip(input_ids_list, attention_mask_list):
                pad_len = longest - len(ids)
                if pad_len > 0:
                    if self.padding_side == "right":
                        ids = ids + [self.pad_token_id] * pad_len
                        mask = mask + [0] * pad_len
                    else:
                        ids = [self.pad_token_id] * pad_len + ids
                        mask = [0] * pad_len + mask
                final_ids.append(ids)
                final_mask.append(mask)
            input_ids_list = final_ids
            attention_mask_list = final_mask
        if return_tensors == "pt":
            try:
                return {
                    "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
                    "attention_mask": torch.tensor(
                        attention_mask_list, dtype=torch.long
                    ),
                }
            except ValueError:
                return {
                    "input_ids": input_ids_list,
                    "attention_mask": attention_mask_list,
                }
        return {"input_ids": input_ids_list, "attention_mask": attention_mask_list}

    def decode(self, token_ids, skip_special_tokens=False):
        res = ""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, int):
            token_ids = [token_ids]
        for i in token_ids:
            idx = i.item() if hasattr(i, "item") else i
            char = self.id_to_token.get(idx, "<unk>")
            if skip_special_tokens and char in ["<pad>", "<s>", "</s>"]:
                continue
            res += char
        return res

    def batch_decode(self, sequences, skip_special_tokens=False):
        return [
            self.decode(s, skip_special_tokens=skip_special_tokens) for s in sequences
        ]

    def __len__(self):
        return len(self.chars)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)
        tokenizer_config = {
            "tokenizer_class": "MathTokenizer",
            "model_max_length": self.model_max_length,
            "padding_side": self.padding_side,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
        }
        with open(
            os.path.join(save_directory, "tokenizer_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        special_tokens_map = {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
        }
        with open(
            os.path.join(save_directory, "special_tokens_map.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)


class QATokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        texts = examples["text"]
        prompts = []
        answers = []
        for t in texts:
            if "=" not in t:
                p, a = t, ""
            else:
                p, a = t.split("=", 1)
            prompts.append(p.strip() + "=")
            answers.append(a.strip())
        enc_p = self.tokenizer(prompts, truncation=True, max_length=MAX_LENGTH)
        enc_a = self.tokenizer(answers, truncation=True, max_length=MAX_LENGTH)
        input_ids_batch, labels_batch, attention_mask_batch = [], [], []
        eos_id = self.tokenizer.eos_token_id
        max_prompt_tokens = max(0, MAX_LENGTH - 2)
        for p_ids, a_ids in zip(enc_p["input_ids"], enc_a["input_ids"]):
            p_trim = p_ids[:max_prompt_tokens]
            remaining = MAX_LENGTH - len(p_trim)
            ans_space = max(0, remaining - 1)
            a_trim = a_ids[:ans_space]
            ids = p_trim + a_trim + [eos_id]
            labels = ([-100] * len(p_trim)) + a_trim + [eos_id]
            attn = [1] * len(ids)
            if len(ids) < MAX_LENGTH:
                pad_len = MAX_LENGTH - len(ids)
                ids += [self.tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len
                attn += [0] * pad_len
            input_ids_batch.append(ids)
            labels_batch.append(labels)
            attention_mask_batch.append(attn)
        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": attention_mask_batch,
        }


def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]
    if hasattr(predictions, "ndim") and predictions.ndim == 3:
        pred_ids = np.argmax(predictions, axis=-1)
    else:
        pred_ids = predictions
    shift_preds = pred_ids[..., :-1]
    shift_labels = labels[..., 1:]
    mask = shift_labels != -100
    if mask.sum() == 0:
        return {"token_acc": 0.0, "exact_match": 0.0}
    token_correct = (shift_preds == shift_labels) & mask
    token_acc = token_correct.sum() / np.maximum(mask.sum(), 1)
    per_row_ok = []
    for i in range(shift_labels.shape[0]):
        m = mask[i]
        if m.sum() == 0:
            per_row_ok.append(0.0)
        else:
            per_row_ok.append(float(np.all(shift_preds[i][m] == shift_labels[i][m])))
    exact = float(np.mean(per_row_ok)) if per_row_ok else 0.0
    return {"token_acc": float(token_acc), "exact_match": exact}


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


if __name__ == "__main__":
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MathTokenizer()
    data_files = {
        "train": os.path.join(DATA_DIR, "train.txt"),
        "validation": os.path.join(DATA_DIR, "val.txt"),
    }
    raw_ds = load_dataset("text", data_files=data_files)
    proc = QATokenizer(tokenizer=tokenizer)
    ds = raw_ds.map(proc, batched=True, remove_columns=["text"])
    vocab_size = len(tokenizer)
    bos_token_id = tokenizer.bos_token_id
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=MODEL_DIM,
        intermediate_size=MODEL_DIM * 4,
        num_hidden_layers=N_LAYER,
        num_attention_heads=N_HEAD,
        num_key_value_heads=N_HEAD,
        max_position_embeddings=MAX_LENGTH,
        bos_token_id=bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        rope_theta=10000.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    model = LlamaForCausalLM(cfg)
    model.resize_token_embeddings(vocab_size)
    model = model.to(torch.float32)
    model.to(device)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        disable_tqdm=False,
        num_train_epochs=3200,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=100,
        learning_rate=5e-6,
        weight_decay=0.0,
        warmup_ratio=0.3,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=torch.cuda.is_available(),
        max_grad_norm=1.0,
        report_to="none",
        load_best_model_at_end=False,
        dataloader_num_workers=0,
        save_total_limit=None,
        seed=SEED,
        eval_accumulation_steps=1,
    )
    trainer = Trainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()
    print("Training completed.")
    print("Evaluating model...")
    metrics = trainer.evaluate()
    print(metrics)
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
