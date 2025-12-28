# train_t5.py
import argparse
import json
import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import sacrebleu

# 配置日志格式，保持和你之前的风格一致
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranslationDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_src_len=128, max_tgt_len=128, prefix="translate Chinese to English: "):
        self.data = []
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.prefix = prefix
        
        logger.info(f"Loading data from {jsonl_path} ...")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        logger.info(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # T5 需要任务前缀
        src_text = self.prefix + item["zh"]
        tgt_text = item["en"]

        # Tokenize inputs
        model_inputs = self.tokenizer(
            src_text, 
            max_length=self.max_src_len, 
            truncation=True,
            padding=False # Padding 由 DataCollator 处理
        )

        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_text, 
                max_length=self.max_tgt_len, 
                truncation=True,
                padding=False
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # --- 修复核心：确保 ID 在合法范围内 ---
    # T5 的 vocab size 是 32128，超过这个范围的 ID 会导致 IndexError
    # 将 -100 替换为 pad_token_id (0)
    # 将任何 >= vocab_size 的 ID 也替换为 pad_token_id (防止越界)
    vocab_size = tokenizer.vocab_size
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    preds = np.where(preds < vocab_size, preds, tokenizer.pad_token_id) 

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = np.where(labels < vocab_size, labels, tokenizer.pad_token_id)
    # -------------------------------------
    
    # 解码生成结果
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 简单的后处理
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    # 计算 BLEU
    result = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)
    return {"bleu": result.score}
    
def main():
    parser = argparse.ArgumentParser()
    # 数据路径：注意这里读取的是 jsonl 原始文件
    parser.add_argument("--train_file", type=str, default="Data/raw/train_100k.jsonl")
    parser.add_argument("--valid_file", type=str, default="Data/raw/valid.jsonl")
    parser.add_argument("--run_dir", type=str, default="runs/t5_finetune")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="t5-small", help="t5-small or t5-base")
    
    # 训练超参
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--accum_steps", type=int, default=1)
    
    args = parser.parse_args()

    # 1. 加载 Tokenizer 和 模型
    logger.info(f"Loading pretrained model: {args.model_name}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # 2. 准备数据集
    train_dataset = TranslationDataset(args.train_file, tokenizer, args.max_len, args.max_len)
    valid_dataset = TranslationDataset(args.valid_file, tokenizer, args.max_len, args.max_len)
    
    # 3. 设置 Data Collator (自动 padding)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 4. 训练参数配置
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.run_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2, # 验证集 batch 可以大一点
        gradient_accumulation_steps=args.accum_steps,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True, # 验证时生成文本计算 BLEU
        fp16=torch.cuda.is_available(),
        generation_max_length=args.max_len,
        logging_dir=f"{args.run_dir}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to="none" # 不上传 wandb，保持简单
    )

    # 5. 初始化 Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=lambda preds: compute_metrics(preds, tokenizer),
        tokenizer=tokenizer,
    )

    # 6. 开始训练
    logger.info("Starting training...")
    trainer.train()
    
    # 7. 保存最终模型
    save_path = os.path.join(args.run_dir, "final_model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()