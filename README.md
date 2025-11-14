# 🔍 Hybrid Multi-Model Subjective Answer Grading System  
A transparent, rubric-driven, human-in-the-loop grading framework combining LLMs, extractive QA models, and sentence-embedding models.

---

## 🌟 Overview  
Modern LLMs can “grade” answers when prompted directly, but their evaluation process is opaque, inconsistent, and prone to hallucination.  
This project builds a **transparent, interpretable, and modular grading pipeline** for subjective answers — designed for real academic settings.

Instead of letting a single LLM judge the entire answer, we break the task into clear, controllable steps:
1. **Rubric Generation**  
2. **Answer Segmentation (model-selectable)**  
3. **AI Tentative Scoring**  
4. **Instructor Validation**  
5. **Rubric Refinement**

This ensures:
- zero hidden reasoning,  
- evidence-based scoring,  
- instructor control over the final marks,  
- support for multiple model endpoints to balance cost and performance.

---

## 🏗️ System Architecture  
<img width="1553" height="731" alt="image" src="https://github.com/user-attachments/assets/a92d2514-f8ec-40f9-bde3-4107fe9dba49" />

---

## ⚙️ Features  
### ✔️ **Rubric Generation (GPT-OSS 120B)**  
Generates clean, structured rubrics using a strict format that downstream modules can reliably parse.

### ✔️ **Three Segmentation Endpoints**  
Choose the model that best suits cost, accuracy, or deployment constraints:

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **GPT-OSS 120B** | Deep semantic understanding, can combine scattered evidence | API cost, occasional paraphrasing |
| **DeBERTa-v3 QA** | Precise extractive spans, CPU-friendly | Only single contiguous span |
| **MPNet Embeddings** | Local, fast, multi-sentence retrieval | Returns full sentences only; may include loosely related ones |

### ✔️ **AI Tentative Scoring**  
LLM assigns provisional marks with explicit evidence for every rubric point.

### ✔️ **Human-in-the-Loop Interface**  
Built in **Streamlit**, enabling instructors to:
- view extracted evidence  
- highlight relevant text in the full answer  
- modify rubric points  
- override scores  
- accept/reject rubric refinements  

### ✔️ **Rubric Refinement Engine**  
Suggests minimal rubric adjustments when students bring up valid but uncovered points.

---

## 🧠 Model Stack  
**Generative Models**
- GPT-OSS 120B (via Groq)
- llama-3.3-70b-versatile (via Groq)

**Extractive / Embedding Models**
- `deberta-v3-large-squad2` (QA)
- `all-mpnet-base-v2` (Sentence Embeddings)

All local models run **entirely on CPU**, reducing the cost of large-scale deployment.

---

---
## Fine tuned Models
You can find the fine tuned model on https://drive.google.com/drive/folders/1lO9oG2EndQOFuoXCRbsD84VdLLF7NGs6?usp=drive_link 

## Datasets 
1. SQuAD 2.0 - https://www.kaggle.com/datasets/thedevastator/squad2-0-a-challenge-for-question-answering-syst
2. HotpotQA - https://www.kaggle.com/datasets/jeromeblanchet/hotpotqa-question-answering-dataset
3. MASHQA - https://drive.google.com/file/d/1RY_gWB4gaUPkW3w9WhIZAwxg5dzNFliK/view

---

