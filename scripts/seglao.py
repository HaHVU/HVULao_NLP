#!/usr/bin/env python3
# --- Thông báo khởi động ---
print("🚀 Starting Lao word segmentation... Please wait!")
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import argparse

# --- Xử lý đối số dòng lệnh ---
parser = argparse.ArgumentParser(description="Tách từ tiếng Lào")
parser.add_argument("-i", "--input", required=True, help="Đường dẫn tới file đầu vào")
parser.add_argument("-o", "--output", required=True, help="Đường dẫn tới file đầu ra")
args = parser.parse_args()

input_file = args.input
output_file = args.output

# --- Thông báo chờ trước khi kiểm tra GPU ---
print("🔄 Please wait, checking available device...")

# --- Kiểm tra GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng: {device}")

# Thông báo cho người dùng biết là đang load mô hình
print("🔄 Please wait, loading model and tokenizer...")

# --- Tải tokenizer và mô hình đã fine-tune ---
model_path = "d:/DatasegLao/lao_finetunedMbert_10k"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
model.eval()

# --- Hàm tách từ tiếng Lào ---
def segment_sentence(sentence):
    tokens = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**tokens)

    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()
    input_ids = tokens["input_ids"][0].cpu().tolist()
    tokens_decoded = tokenizer.convert_ids_to_tokens(input_ids)

    word_start_labels = {id for id, label in model.config.id2label.items() if label == "B-WORD"}
    segmented_sentence = []
    current_word = ""

    for token, pred in zip(tokens_decoded, predictions):
        if token in ["<s>", "</s>", "<pad>", "<unk>"]:
            continue  # Bỏ qua token đặc biệt

        clean_token = token.lstrip("▁")

        if pred in word_start_labels or token.startswith("▁"):
            if current_word:
                segmented_sentence.append(current_word)
            current_word = clean_token
        elif pred == 2:
            current_word += clean_token
        else:
            if current_word:
                segmented_sentence.append(current_word)
            segmented_sentence.append(clean_token)
            current_word = ""

    if current_word:
        segmented_sentence.append(current_word)

    return " ".join(segmented_sentence)

# --- Đọc file đầu vào và ghi kết quả ra file đầu ra ---
with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        line = line.strip()
        if line:
            segmented_line = segment_sentence(line)
            f_out.write(segmented_line + "\n")

print(f"✅ Quá trình tách từ hoàn tất! Kết quả được lưu vào {output_file}")
