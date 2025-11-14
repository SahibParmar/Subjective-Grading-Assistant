from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer, util
import nltk
import torch

GROQ_MODEL_NAME="openai/gpt-oss-120b" #"llama-3.3-70b-versatile"


# Load .env file
load_dotenv()

def use_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY in .env file")

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    response = model.invoke("Hello, world!")
    return response


def use_groq(prompt,model_name=GROQ_MODEL_NAME):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in .env file")

    model = ChatGroq(model=model_name, groq_api_key=api_key)
    
    response = model.invoke(prompt)
    return response


def use_deberta(answer, rubric_dict):
    """
    Uses RoBERTa QA model to extract relevant answer segments for each rubric point.
    Returns text in the same <start>...<end> format for parser compatibility.
    """

    model_name = "deepset/deberta-v3-large-squad2"#"deepset/roberta-large-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    qa_pipeline = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        handle_impossible_answer=False,
        max_answer_len=200,
    )

    formatted_output = ["<start>"]

    for rubric_point, _ in rubric_dict.items():
        try:
            qa_input = {"question": rubric_point, "context": answer}
            res = qa_pipeline(qa_input)
            extracted = res.get("answer", "").strip()
            if not extracted: #or res.get("score", 0) < 0.1:
                extracted = "Not addressed"
        except Exception:
            extracted = "Not addressed"

        formatted_output.append(f"    Rubric: {rubric_point}")
        formatted_output.append(f"    corresponding_part: {extracted}")
        formatted_output.append("    ####")

    formatted_output.append("<end>")
    return "\n".join(formatted_output)

def extract_relevant_passages(answer, rubric_dict, top_k=3):
    # nltk.download('punkt')
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    model = SentenceTransformer('all-mpnet-base-v2')
    sentences = nltk.sent_tokenize(answer)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    
    output = ["<start>"]
    for rubric_point in rubric_dict.keys():
        rubric_embedding = model.encode(rubric_point, convert_to_tensor=True)
        cosine_scores = util.cos_sim(rubric_embedding, sentence_embeddings)[0]
        
        top_indices = cosine_scores.topk(top_k).indices.tolist()
        relevant_parts = " ".join([sentences[i] for i in top_indices])
        
        output.append(f"    Rubric: {rubric_point}")
        output.append(f"    corresponding_part: {relevant_parts}")
        output.append("    ####")
    output.append("<end>")
    return "\n".join(output)
def extract_relevant_passages_2(answer, rubric_dict, top_k=3):
    """
    Improved version: ensures that each sentence in the student's answer
    is assigned to at most one rubric point.
    """

    # Ensure sentence tokenizer availability
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    # Initialize model and encode once
    model = SentenceTransformer('all-mpnet-base-v2')
    sentences = nltk.sent_tokenize(answer)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    used_mask = torch.zeros(len(sentences), dtype=torch.bool)
    output = ["<start>"]

    for rubric_point in rubric_dict.keys():
        rubric_embedding = model.encode(rubric_point, convert_to_tensor=True)
        cosine_scores = util.cos_sim(rubric_embedding, sentence_embeddings)[0]

        # Mask already-used sentences
        cosine_scores[used_mask] = -1e9

        # Get top-k available sentences
        top_indices = cosine_scores.topk(min(top_k, (~used_mask).sum())).indices.tolist()

        # Mark selected as used
        used_mask[top_indices] = True

        relevant_parts = " ".join([sentences[i] for i in top_indices]) if top_indices else "Not addressed"

        output.append(f"    Rubric: {rubric_point}")
        output.append(f"    corresponding_part: {relevant_parts}")
        output.append("    ####")

    output.append("<end>")
    return "\n".join(output)

if __name__ == "__main__":
    question="write 10 lines on generative AI"
    marks=10

    prompt = f"""
    You are an expert educational evaluator and assessment designer.

    Your task is to generate a **grading rubric** for the following question:

    Question:
    \"\"\"{question}\"\"\"
    Maximum marks: {marks}

    ### Instructions:
    1. Identify the key concepts, reasoning steps, or elements that a high-quality answer must contain.
    2. Divide the rubric into **clear sub-criteria**, each worth some marks (the total should sum to {marks}).
    3. Each criterion should have:
    - Criterion Name
    - Description of what to look for in the student's answer
    - Marks allocated
    - Examples of what constitutes full, partial, or no credit for that criterion.
    4. If the question is open-ended, include one criterion for clarity, coherence, and relevance.


    ### Additional Notes:
    - Ensure the rubric is **balanced** (no single criterion dominates the total marks).
    - Prefer **conceptual evaluation** over rote memorization.
    - Do not repeat the question in the rubric.
    - just give rubric points. nothing else.

    you should give rubric in below format only:
    Rubric: <rubric point>
    Marks: <marks for the rubric point>
    ####
    Rubric: <rubric point>
    Marks: <marks for the rubric point>
    ####
    Rubric: <rubric point>
    Marks: <marks for the rubric point>
    ####

    and so on

    Now, generate the rubric.
    """
    #    5. Present the rubric in **structured JSON** format as shown below.

    # Output Format (JSON):
    # {{
    # "total_marks": 10,
    # "criteria": [
    #     {{
    #     "name": "Concept Understanding",
    #     "description": "Checks if the student demonstrates understanding of key principles.",
    #     "marks": 3,
    #     "grading_examples": {{
    #         "full_credit": "Accurately explains all key concepts with correct terminology.",
    #         "partial_credit": "Mentions most key ideas but with minor inaccuracies.",
    #         "no_credit": "Fails to demonstrate understanding of the concept."
    #     }}
    #     }},
    #     ...
    # ]
    # }}
    response = use_groq(prompt)
    print("Generated Grading Rubric:")
    print(response.content)