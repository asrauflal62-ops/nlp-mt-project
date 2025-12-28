# decode_t5.py
import argparse
import json
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sacrebleu

# 复用 train_t5.py 中的 Dataset 类，但稍微简化
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, prefix="translate Chinese to English: "):
        self.data = []
        self.prefix = prefix
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the finetuned model folder")
    parser.add_argument("--data_file", type=str, default="Data/raw/test.jsonl")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--beam", type=int, default=1, help="Beam size, 1 for greedy")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 1. 加载模型 (自动寻找 run_dir 下的 checkpoint 或 final_model)
    # 优先加载 final_model，如果不存在则加载 checkpoint-best (HuggingFace 默认保存结构)
    model_path = os.path.join(args.run_dir, "final_model")
    if not os.path.exists(model_path):
        # 尝试寻找 checkpoint 文件夹
        subdirs = [d for d in os.listdir(args.run_dir) if d.startswith("checkpoint")]
        if subdirs:
            # 找最新的 checkpoint
            subdirs.sort(key=lambda x: int(x.split("-")[-1]))
            model_path = os.path.join(args.run_dir, subdirs[-1])
            print(f"Loading from checkpoint: {model_path}")
        else:
            raise FileNotFoundError(f"No model found in {args.run_dir}")
    else:
        print(f"Loading from final model: {model_path}")

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(args.device)
    model.eval()

    # 2. 准备数据
    dataset = InferenceDataset(args.data_file)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)

    preds = []
    refs = []
    
    print(f"Start decoding on {args.data_file}...")
    
    # 3. 推理循环
    for batch in tqdm(loader):
        # 准备输入
        src_texts = [dataset.prefix + item for item in batch["zh"]]
        tgt_texts = batch["en"] # 参考答案
        
        inputs = tokenizer(
            src_texts, 
            max_length=args.max_len, 
            truncation=True, 
            padding=True, 
            return_tensors="pt"
        ).to(args.device)

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=args.max_len,
                num_beams=args.beam,
                early_stopping=(args.beam > 1)
            )

        # 解码
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        preds.extend([p.strip() for p in batch_preds])
        refs.extend([[t.strip()] for t in tgt_texts]) # sacrebleu 需要 list of list

    # 4. 计算 BLEU
    bleu = sacrebleu.corpus_bleu(preds, refs)
    print(f"BLEU: {bleu.score:.2f}")

    # 5. 保存结果 (保持和你之前一样的格式)
    out_dir = os.path.join(args.run_dir, f"eval_t5_beam{args.beam}")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "pred.txt"), "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p + "\n")
            
    with open(os.path.join(out_dir, "ref.txt"), "w", encoding="utf-8") as f:
        for r in refs:
            f.write(r[0] + "\n") # refs 是 [[ref1], [ref2]]
            
    with open(os.path.join(out_dir, "samples.txt"), "w", encoding="utf-8") as f:
        for i in range(min(10, len(preds))):
            f.write(f"Pred: {preds[i]}\nRef:  {refs[i][0]}\n\n")
            
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()