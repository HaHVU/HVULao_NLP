readme_content = """
# HVULao_NLP

**HVULao_NLP** is a project dedicated to sharing datasets and tools for Lao Natural Language Processing (NLP), developed and maintained by the research team at **Hung Vuong University (HVU), Phu Tho, Vietnam**.  
This project is supported by **Hung Vuong University, Phu Tho, Vietnam**, with the aim of advancing research and applications in low-resource language processing, particularly for the Lao language.

## 📁 Datasets

This repository provides a semi-automatically constructed corpus consisting of Lao sentences that have been **word-segmented** and **part-of-speech (POS) tagged**. It is designed to support a wide range of NLP applications, including language modeling, sequence labeling, linguistic research, and the development of Lao language tools.

### 🔹 `Datatest1k/`

This folder contains the **test dataset**, consisting of 1,000 Lao sentences:

- **`testorgin1k.txt`**  
  → The original 1,000 Lao sentences in raw form (without segmentation or annotation).

- **`testtag1k.json`**  
  → The same 1,000 sentences with **word segmentation** and **POS tagging**. These sentences are created using large language models (LLMs) like ChatGPT and then manually reviewed and corrected by native linguists.

### 🔹 `Datatrain10k/`

This folder contains the **training dataset**, consisting of 10,000 Lao sentences:

- **`10ktrainorin.txt`**  
  → The original 10,000 Lao sentences in raw form (without segmentation or annotation).

- **`10ktraintag.json`**  
  → The same 10,000 sentences with **word segmentation** and **POS tagging**, created using the same method as the test data.

> All data files are UTF-8 encoded and prepared for easy use in NLP pipelines.

## 📁 The Lao sentence segment tool:

This is a command-line tool for Lao word segmentation using a fine-tuned `transformers` model. It leverages Hugging Face's `transformers` library and PyTorch for efficient token classification.

---

## Features

- Lao language word segmentation using a pre-trained and fine-tuned model
- Simple command-line usage
- GPU support for faster processing (if available)

---

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- A fine-tuned model (Please download at: https://huggingface.co/Tienha123/Segmenttool/tree/main`)

You can install the dependencies with:

```bash
pip install torch transformers
```

---

## Setup

1. Clone this repository (if from GitHub or Hugging Face).
2. Place your fine-tuned model folder in the same directory or update the `model_path` in the script accordingly.
3. Make sure your Python script is executable:

   ```bash
   chmod +x segment_lao.py
   ```

---

## Input Format

- Input should be a **UTF-8 encoded text file**.
- Each line should contain a Lao sentence (without tokenization).

---

## Usage

Run the script from the command line:

```bash
python3 segment_lao.py -i <input_file> -o <output_file>
```

### Arguments

- `-i`, `--input`: Path to the input file (required)
- `-o`, `--output`: Path to the output file (required)

### Example

```bash
python3 segment_lao.py -i ./data/lao_raw.txt -o ./output/lao_segmented.txt
```

---

## How It Works

- Loads a fine-tuned token classification model (`lao_finetuned_10k`)
- Performs word segmentation based on B-WORD label prediction
- Outputs segmented sentences to the specified output file

---

## Device Support

The tool automatically uses **GPU** if available. If not, it falls back to **CPU**.

---

## Sample Output

Input:

```bash
ຂ້ອຍຮັກພາສາລາວ
```

Output:

```bash
ຂ້ອຍ ຮັກ ພາສາ ລາວ
```

---

## 📁 The Lao sentence POS tagging tool:
## Installing CRF++ 0.58 on Windows

### Step 1: Download and Extract

- Visit: [https://github.com/taku910/crfpp/releases](https://github.com/taku910/crfpp/releases)
- Download the file: `CRF++-0.58.zip`
- Extract it to a folder, for example: `D:\your\directory\CRF++-0.58`

### Step 2: Compile CRF++ (if needed)

**Simple method:** If the extracted folder already contains executables like `crf_test.exe`, you can use them directly. Otherwise, you need to:

1. Install **MinGW** or **MSYS2** to get the `make` command  
2. Open a terminal and run the following commands:

   ```bash
   cd D:/KLTN/CRF++-0.58
   ./configure
   make
   ```

### Step 3: Add CRF++ to the `Path` Environment Variable

1. Open the **Start menu** → search for `Edit the system environment variables`
2. In the **System Properties** window → click **Environment Variables**
3. Under **System variables**, select the entry named **Path**, then click **Edit**
4. Click **New**, then enter the path to the folder containing `crf_test.exe`, for example:

   ```bash
   D:/KLTN/CRF++-0.58
   ```

5. Click OK and exit

### Step 4: Verify Installation

1. Open **Command Prompt (cmd)**
2. Enter the following command:

   ```bash
   crf_test -h
   ```

3. If the help message appears → **CRF++ has been successfully installed**

---

## Installing CRF++ 0.58 in Visual Studio Code

### Steps 1, 2, and 3 are the same as the installation on Windows

### Step 4: Verify

1. Open a **New Terminal**
2. Enter the following command:

   ```bash
   crf_test -h
   ```

3. If the help message appears → **CRF++ environment has been successfully initialized**

---

## Installing CRF++ 0.58 on Linux/macOS

### Step 1: Install Required Packages

For **Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install g++ make automake
```

For **macOS**

```bash
xcode-select --install
```

### Step 2: Download CRF++ from GitHub

```bash
git clone https://github.com/taku910/crfpp
cd crfpp
```

### Step 3: Compile and Install

```bash
./configure
make
sudo make install
```

### Step 4: Verify installation

```bash
crf_learn --version
```

### If you encounter a "Command not found"

Make sure **/usr/local/bin** is included in your **$PATH**
Add the following line to your **~/.bashrc** or **~/.zshrc**

```bash
export PATH=/usr/local/bin:$PATH
```

Then reload the shell:

```bash
source ~/.bashrc   # or: source ~/.zshrc
```

---

## How to Run the LaoPostagtool
- The first, you have to download the model from this link: https://huggingface.co/Tienha123/LaoPostagtool/tree/main
- after, extract it to root folder.
- Move into this folder

- In VS Code: Open Terminal and run the following command:

```bash
python3 Pos_tagging.py <segmented_input_file> <output_directory>
```

**python3** is used to execute the script using Python 3.

**Pos_tagging.py** is the Python script used for part-of-speech tagging. Make sure this file is in the current directory or provide the correct path

**segmented_input_file** is the path to the input file containing tokenized (segmented) sentences. Each line in the file should be a tokenized sentence.
Example: ໂດຍການ ຖືວ່າ ລັດຖະບານ ເປັນ ສິ່ງທີ່ ຫລີກລ້ຽງ ບໍ່ໄດ້

**output_directory** is the folder where preprocessed and POS-tagged output files will be saved.

**Example**  

```bash
python3 ./Pos_tagging.py ./Test/lao_sentences_segmented.txt Test1
```

**.\Test\lao_sentences_segmented.txt:** is the path to the tokenized input file
**Test1:** is the output directory where the POS-tagged results will be stored`

## 📚 Usage

These datasets are intended for:

- Training and evaluating sequence labeling models (e.g., CRF, BiLSTM, mBERT)
- Developing Lao NLP tools (e.g., POS taggers, tokenizers)
- Conducting linguistic research on Lao

## 🤝 Contribution

We welcome contributions! Feel free to:

- Open issues
- Submit pull requests
- Suggest improvements or additional data

## 📄 Citation

If you use this repository or our dataset in your research, please cite the following paper:
```Cite
**Ha Nguyen-Tien**, Thongphan Palongve, Cuong Nguyen-Quy, and Kien Le-Trung.  
"A Novel Approach to Building Word Segmentation and POS-Tagging Corpora for Low-Resource Languages."  
Faculty of Engineering and Technology, Hung Vuong University, Nguyen Tat Thanh Street, Phu Tho 350000, Vietnam.  
Published in IEEE Access.  
Date of publication: xxxx 00, 0000  
Date of current version: xxxx 00, 0000  
DOI: [10.1109/ACCESS.2025.0429000](https://doi.org/10.1109/ACCESS.2025.0429000)  
Corresponding author: **Ha Nguyen-Tien** (nguyentienha@hvu.edu.vn)
```

### 📚 BibTeX

```bibtex
@article{nguyen2024wordsegmentation,
  author    = {Nguyen-Tien, Ha and Palongve, Thongphan and Nguyen-Quy, Cuong and Le-Trung, Kien},
  title     = {A Novel Approach to Building Word Segmentation and POS-Tagging Corpora for Low-Resource Languages},
  journal   = {IEEE Access},
  year      = {2025},
  doi       = {10.1109/ACCESS.2025.0429000},
  note      = {Faculty of Engineering and Technology, Hung Vuong University, Vietnam}
}
```

## 📬 Contact

- **Ha Nguyen-Tien** (Corresponding author)  
  Email: nguyentienha@hvu.edu.vn

- **Thongphan Palongve**  
  Email: tkvue73@gmail.com

- **Cuong Nguyen-Quy**  
  Email: t12345cuong@gmail.com

- **Kien Le-Trung**  
  Email: kiendeptrai06022003@gmail.com

📍 Institution: Faculty of Engineering and Technology, Hung Vuong University, Phu Tho, Vietnam  
🌐 Website: [https://hvu.edu.vn](https://hvu.edu.vn)

---

*This repository is part of our ongoing effort to support Lao NLP and make language technology more accessible for underrepresented and low-resource languages.*
"""
