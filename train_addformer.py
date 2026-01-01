import os
import inspect
import warnings
import re
from dataclasses import dataclass
from typing import Optional, List, Dict
import json
import gc
import numpy as np
import torch
import matplotlib.pyplot as plt
import multiprocessing as mp
import queue
import time
from datasets import load_dataset
from tqdm.auto import tqdm
import matplotlib.ticker as ticker
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    LlamaConfig,
    LlamaForCausalLM,
    TrainerCallback,
)
from transformers.trainer import Trainer as HFTrainer
from transformers import logging as hf_logging
from datasets import logging as ds_logging

hf_logging.set_verbosity_error()
ds_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

DATA_DIR = "data"
OUTPUT_DIR = "addformer_ckpt"
MODEL_DIM = 256
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
    if isinstance(eval_pred, tuple) and len(eval_pred) >= 2:
        predictions, labels = eval_pred[0], eval_pred[1]
    else:
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


def _plot_process_worker(data_queue, output_dir):
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=False, constrained_layout=True
    )

    try:
        fig.canvas.manager.set_window_title("Training Monitor (Multiprocess)")
    except Exception:
        pass

    ax1.set_title("Training Loss Curve")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    (line_loss,) = ax1.plot([], [], "r-", linewidth=1.5, label="Loss")
    ax1.legend(loc="upper right")

    ax2.set_title("Learning Rate Schedule")
    ax2.set_ylabel("Learning Rate")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    (line_lr,) = ax2.plot([], [], "b-", linewidth=1.5, label="LR")
    ax2.legend(loc="upper right")

    steps = []
    losses = []
    lrs = []

    running = True
    while running:
        try:
            while not data_queue.empty():
                item = data_queue.get_nowait()
                if item is None:
                    running = False
                    break

                step, loss, lr = item
                steps.append(step)
                losses.append(loss)
                lrs.append(lr)

            if steps:
                line_loss.set_data(steps, losses)
                line_lr.set_data(steps, lrs)

                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()

                fig.canvas.draw()
                fig.canvas.flush_events()

        except Exception:
            pass

        plt.pause(0.5)

    plt.close(fig)


class DynamicPlotCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.queue = mp.Queue()
        self.process = mp.Process(
            target=_plot_process_worker, args=(self.queue, self.output_dir)
        )
        self.process.daemon = True
        self.process.start()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = state.global_step
            loss = logs["loss"]
            lr = logs.get("learning_rate", 0.0)
            self.queue.put((step, loss, lr))

    def on_train_end(self, args, state, control, **kwargs):
        self.queue.put(None)
        if self.process.is_alive():
            self.process.join(timeout=5)


class MemSafeTrainer(HFTrainer):
    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs
    ):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        total_loss = 0.0
        total_token_correct = 0
        total_token_masked = 0
        total_exact_matches = 0
        total_samples = 0
        total_steps = 0

        model = self.model
        model.eval()

        for step, batch in enumerate(
            tqdm(eval_dataloader, desc="Evaluating", leave=False)
        ):
            batch = {k: v.to(self.args.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1)
            labels = batch["labels"]

            shift_preds = pred_ids[..., :-1]
            shift_labels = labels[..., 1:]

            mask = shift_labels != -100

            token_correct = (shift_preds == shift_labels) & mask
            total_token_correct += token_correct.sum().item()
            total_token_masked += mask.sum().item()

            for i in range(shift_labels.size(0)):
                row_mask = mask[i]
                if row_mask.sum() > 0:
                    is_exact = (
                        (shift_preds[i][row_mask] == shift_labels[i][row_mask])
                        .all()
                        .item()
                    )
                    total_exact_matches += int(is_exact)
                else:
                    pass

            total_samples += shift_labels.size(0)
            total_steps += 1

            del (
                outputs,
                batch,
                logits,
                pred_ids,
                shift_preds,
                shift_labels,
                mask,
                token_correct,
            )

            if step % 1000 == 0:
                torch.cuda.empty_cache()

        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        avg_token_acc = (
            total_token_correct / total_token_masked if total_token_masked > 0 else 0.0
        )
        avg_exact_match = (
            total_exact_matches / total_samples if total_samples > 0 else 0.0
        )

        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_token_acc": avg_token_acc,
            f"{metric_key_prefix}_exact_match": avg_exact_match,
            f"{metric_key_prefix}_samples": total_samples,
        }

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        gc.collect()
        torch.cuda.empty_cache()

        return metrics


def _find_latest_checkpoint(base_dir: str) -> Optional[str]:
    if not os.path.isdir(base_dir):
        return None
    latest_num = -1
    latest_path = None
    for entry in os.listdir(base_dir):
        match = re.match(r"checkpoint-(\d+)$", entry)
        if match:
            num = int(match.group(1))
            if num > latest_num:
                latest_num = num
                latest_path = os.path.join(base_dir, entry)
    return latest_path


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

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

    latest_checkpoint = _find_latest_checkpoint(OUTPUT_DIR)

    if latest_checkpoint:
        print(f"Resuming from: {latest_checkpoint}")
        model = LlamaForCausalLM.from_pretrained(latest_checkpoint)
    else:
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
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=100,
        learning_rate=5e-6,
        weight_decay=0.0,
        warmup_ratio=0.1,
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

    trainer = MemSafeTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[DynamicPlotCallback(output_dir=OUTPUT_DIR)],
    )

    if latest_checkpoint:
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()

    trainer.save_model(OUTPUT_DIR)
