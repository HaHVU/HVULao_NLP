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

## 📬 Contact

- **Maintainer**: Nguyen Tien Ha  
- **Email**: nguyentienha@hvu.edu.vn  
- **Institution**: Hung Vuong University, Phu Tho, Vietnam  
- **Website**: [https://hvu.edu.vn](https://hvu.edu.vn)

---

*This repository is part of our ongoing effort to support Lao NLP and make language technology more accessible for underrepresented and low-resource languages.*
"""
