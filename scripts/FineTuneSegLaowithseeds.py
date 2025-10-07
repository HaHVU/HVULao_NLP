# FineTuneSegLaowithseed.py
# Train + Evaluate nhiều seed, GIỮ các hyper-params cốt lõi; bỏ tham số dễ xung đột phiên bản.
import os, json
from statistics import mean, pstdev
import torch
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          Trainer, TrainingArguments)
from transformers.trainer_utils import set_seed
from datasets import Dataset

# ===== Đường dẫn TRÊN MÁY CHỦ =====
TRAIN_JSON = "./Model10000.JSON"     # train 10k (input+output)
TEST_JSON  = "./testtag1k.json"      # test 1k  (input+output)
OUT_ROOT   = "/workspace/Seglao_runs/results_multi_seed"
LOG_ROOT   = "/workspace/Seglao_runs/logs_multi_seed"
EVAL_ROOT  = "/workspace/Seglao_runs/eval_multi_seed"
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(LOG_ROOT, exist_ok=True)
os.makedirs(EVAL_ROOT, exist_ok=True)

SEEDS = [13, 21, 42, 87, 123]

print("Đang tải tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
label2id = {"O": 0, "B-WORD": 1, "I-WORD": 2}
id2label = {v: k for k, v in label2id.items()}
print(f"Nhãn sử dụng: {label2id}")

with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    train_data_all = json.load(f)
with open(TEST_JSON, "r", encoding="utf-8") as f:
    test_data_all = json.load(f)

def align_labels(input_text, output_text):
    enc = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_offsets_mapping=True,
    )
    words = output_text.split()
    boundaries, pos = [], 0
    for w in words:
        idx = input_text.find(w, pos)
        if idx == -1:
            idx = pos
        boundaries.append(idx)
        pos = idx + len(w)

    labels = []
    for off in enc["offset_mapping"]:
        if off == (0, 0):
            labels.append(0)  # GIỮ NGUYÊN (pad/special -> O) như paper gốc
            continue
        s, _ = off
        labels.append(label2id["B-WORD"] if s in boundaries else label2id["I-WORD"])
    enc.pop("offset_mapping")
    return enc, labels

def tokenize_function(ex):
    toks, labs = align_labels(ex["input"], ex["output"])
    toks["labels"] = labs
    return toks

print("Đang chuẩn bị dataset...")
train_ds_full = Dataset.from_list(train_data_all).map(tokenize_function)
split = train_ds_full.train_test_split(test_size=0.15, seed=42)  # cố định split như paper
train_ds, val_ds = split["train"], split["test"]
print(f"Số lượng mẫu train: {len(train_ds)}, validation: {len(val_ds)}")

# ==== Đánh giá đúng như bạn dùng trong paper (word-set) ====
def calc_precision_recall_wordset(true_sentences, pred_sentences):
    total_tp = total_fp = total_fn = 0
    for true_text, pred_text in zip(true_sentences, pred_sentences):
        true_words = set(true_text.split())
        pred_words = set(pred_text.split())
        tp = len(true_words & pred_words)
        fp = len(pred_words - true_words)
        fn = len(true_words - pred_words)
        total_tp += tp; total_fp += fp; total_fn += fn
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    return precision, recall, f1

def reconstruct_segmented_sentence(input_text, pred_label_ids, max_length=128):
    enc = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
        return_attention_mask=True,
    )
    offsets = enc["offset_mapping"]
    attn    = enc["attention_mask"]

    words = []
    cur_start = None
    prev_end  = None
    for i, (off, m) in enumerate(zip(offsets, attn)):
        if m == 0:
            break
        if off == (0, 0):
            continue
        s, e = off
        if e <= s:
            continue
        lab = int(pred_label_ids[i])
        if lab == label2id["B-WORD"] or cur_start is None:
            if cur_start is not None and prev_end is not None:
                words.append(input_text[cur_start:prev_end])
            cur_start = s
        prev_end = e
    if cur_start is not None and prev_end is not None:
        words.append(input_text[cur_start:prev_end])

    words = [" ".join(w.split()) for w in words if w.strip() != ""]
    return " ".join(words)

def segment_with_model(model_dir, raw_inputs, batch_size=16, max_length=128, device=None):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForTokenClassification.from_pretrained(model_dir)
    mdl.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)

    preds_all = []
    with torch.no_grad():
        for st in range(0, len(raw_inputs), batch_size):
            batch = raw_inputs[st:st+batch_size]
            enc = tok(batch, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = mdl(**enc).logits
            ids = torch.argmax(logits, dim=-1).cpu().numpy()
            for raw, lab_ids in zip(batch, ids):
                preds_all.append(reconstruct_segmented_sentence(raw, lab_ids, max_length=max_length))
    return preds_all

raw_test  = [x["input"].strip()  for x in test_data_all]
gold_test = [x["output"].strip() for x in test_data_all]

eval_csv = os.path.join(EVAL_ROOT, "eval_seed_results.csv")
with open(eval_csv, "w", encoding="utf-8") as f:
    f.write("seed,precision,recall,f1\n")

per_p, per_r, per_f1 = [], [], []

for sd in SEEDS:
    print(f"\n========== HUẤN LUYỆN VỚI SEED = {sd} ==========")
    set_seed(sd)

    model = AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    out_dir = os.path.join(OUT_ROOT, f"seed_{sd}")
    log_dir = os.path.join(LOG_ROOT, f"seed_{sd}")
    os.makedirs(out_dir, exist_ok=True)

    # >>> Tối giản TrainingArguments để tránh key lạ
    args = TrainingArguments(
        output_dir=out_dir,
        do_train=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.02,
        logging_dir=log_dir,
        logging_steps=50,   # chỉ log bước, không yêu cầu eval trong loop
        fp16=True,
        warmup_ratio=0.1,
        seed=sd,
        data_seed=sd,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,   # có thể không dùng trong loop nhưng để đây không hại
        # Không truyền EarlyStoppingCallback ở môi trường hiện tại để tránh lỗi
    )
    trainer.train()

    final_dir = os.path.join(out_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Evaluate theo đúng hàm của bạn
    print(f"Đánh giá seed {sd} …")
    preds = segment_with_model(final_dir, raw_test, batch_size=16, max_length=128)
    with open(os.path.join(EVAL_ROOT, f"lao_segmented_seed{sd}.txt"), "w", encoding="utf-8") as f:
        for line in preds:
            f.write(line + "\n")

    def calc(true_sents, pred_sents):
        return calc_precision_recall_wordset(true_sents, pred_sents)

    p, r, f1 = calc(gold_test, preds)
    print(f"[seed {sd}] P={p:.4f} R={r:.4f} F1={f1:.4f}")
    with open(eval_csv, "a", encoding="utf-8") as f:
        f.write(f"{sd},{p:.4f},{r:.4f},{f1:.4f}\n")
    per_p.append(p); per_r.append(r); per_f1.append(f1)

# Tóm tắt
if per_f1:
    def msd(v): 
        return mean(v), (pstdev(v) if len(v) > 1 else 0.0)
    mp, sp = msd(per_p); mr, sr = msd(per_r); mf, sf = msd(per_f1)
    print("\n=== Tóm tắt (mean ± sd) ===")
    print(f"P = {mp:.4f} ± {sp:.4f}")
    print(f"R = {mr:.4f} ± {sr:.4f}")
    print(f"F1 = {mf:.4f} ± {sf:.4f}")
else:
    print("\n[ERROR] Không có seed nào được đánh giá.")
