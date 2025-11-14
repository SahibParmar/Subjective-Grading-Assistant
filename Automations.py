from Generative_models import use_groq,use_deberta,extract_relevant_passages,extract_relevant_passages_2
import textwrap
from Parsers import parse_answer_segments, parse_rubric,parse_tentative_scores

def generate_rubric(question,marks,endpoint='groq'):
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
      <start>
      Rubric: <rubric point>
      Marks: <marks for the rubric point>
      ####
      Rubric: <rubric point>
      Marks: <marks for the rubric point>
      ####
      Rubric: <rubric point>
      Marks: <marks for the rubric point>
      ####
      <end>

      Now, generate the rubric.
      """
    if endpoint=='groq':
        response = use_groq(prompt)
    else:
        raise ValueError(f'Unsupported endpoint: {endpoint}')
    return response.content


def generate_rubric_2(question, marks, demo_answers="", endpoint='groq'):
    
    # Build the prompt dynamically
    prompt_lines = [
        "You are an expert educational evaluator and assessment designer.",
        "Your task is to generate a **grading rubric** for the following question:",
        f"Question:\n\"\"\"{question}\"\"\"",
        f"Maximum marks: {marks}"
    ]

    # Conditionally add the demo answers if they exist
    if demo_answers and demo_answers.strip():
        prompt_lines.append("\n### Exemplar/Demo Answers (for context):")
        prompt_lines.append(f"\"\"\"{demo_answers}\"\"\"")
        prompt_lines.append("Use these answers to help identify the key concepts and desired depth for the rubric.")

    # Add the rest of the instructions
    prompt_lines.extend([
        "\n### Instructions:",
        "1. Identify the key concepts, reasoning steps, or elements that a high-quality answer must contain.",
        f"2. Divide the rubric into **clear sub-criteria**, each worth some marks (the total should sum to {marks}).",
        "3. Each criterion should have:",
        "   - Criterion Name",
        "   - Description of what to look for in the student's answer",
        "   - Marks allocated",
        "   - Examples of what constitutes full, partial, or no credit for that criterion.",
        "4. If the question is open-ended, include one criterion for clarity, coherence, and relevance.",
        "\n### Additional Notes:",
        "- Ensure the rubric is **balanced** (no single criterion dominates the total marks).",
        "- Prefer **conceptual evaluation** over rote memorization.",
        "- Do not repeat the question in the rubric.",
        "- just give rubric points. nothing else.",
        "\nyou should give rubric in below format only:",
        "<start>",
        "Rubric: <rubric point>",
        "Marks: <marks for the rubric point>",
        "####",
        "Rubric: <rubric point>",
        "Marks: <marks for the rubric point>",
        "####",
        "Rubric: <rubric point>",
        "Marks: <marks for the rubric point>",
        "####",
        "<end>",
        "\nNow, generate the rubric."
    ])

    # Join all lines into the final prompt
    prompt = "\n".join(prompt_lines)
    # --- MODIFICATION END ---

    if endpoint=='groq':
        response = use_groq(prompt)
    else:
        raise ValueError(f'Unsupported endpoint: {endpoint}')
    return response.content




def break_answer_into_points(answer, rubric, endpoint='groq'):
    """
    Classify sections of a student's answer under the given rubric criteria.

    Args:
        answer (str): The student's subjective answer.
        rubric (str): The generated rubric text (from LLM).
        endpoint (str): Which model endpoint to use ('groq' or 'gemini').

    Returns:
        str: LLM-generated structured mapping from rubric → corresponding part.
    """

    classification_prompt = f"""
    You are an expert evaluator and text analyzer.

    You are given:
    1. A grading rubric with multiple rubric points.
    2. A student's subjective answer.

    

    Your task is to map each rubric point to the **most relevant verbatim quote** from the student's answer.
    The quote you extract MUST be **copied exactly, character-for-character,** from the student's answer.
    Do not summarize, rephrase, or change any text.
    If a rubric criterion is not addressed, write 'Not addressed'.

    ### FORMAT STRICTLY REQUIRED:
    <start>
    Rubric: <rubric point>
    corresponding_part: <relevant part from answer>
    ####
    Rubric: <rubric point>
    corresponding_part: <relevant part from answer>
    ####
    ...
    <end>

    ### Input Data
    Rubric:
    {rubric}

    Answer:
    {answer}

    Now generate the structured mapping as per the required format.
    """

    # Clean the prompt (remove unnecessary indentation)
    prompt = textwrap.dedent(classification_prompt).strip()

    # Call the appropriate LLM endpoint
    if endpoint == 'groq':
        response = use_groq(prompt)
    elif endpoint.lower() == 'deberta':
        # raise Exception(type(rubric))
        return use_deberta(answer, rubric)
    elif endpoint=='embedding_model':
        return extract_relevant_passages_2(answer, rubric,top_k=3)
    else:
        raise ValueError("Currently only 'groq' endpoint is supported.")

    return response.content


def ai_grade_segments(answer, rubric, segments, endpoint='groq'):
    """
    Suggests tentative scores for each rubric point based on extracted answer segments.
    Returns a dict {rubric_point: tentative_score}.
    """
    prompt = f"""
    You are an expert teacher grading a student's answer.

    Given the rubric and the extracted answer segments for each criterion,
    assign tentative scores out of the marks allocated.

    Return in this format strictly:
    <start>
    Rubric: <rubric point>
    Tentative_Score: <score>
    ####
    ...
    <end>

    Rubric:
    {rubric}

    Extracted Segments:
    {segments}
    """
    response = use_groq(prompt)
    return parse_tentative_scores(response.content)




def suggest_rubric_modification(answer, rubric, endpoint='groq'):  #, segments
    import textwrap
    from Generative_models import use_groq
#   The AI previously extracted the following mapping of answer parts to rubric points:
#     {segments}
    prompt = f"""
    You are an expert educational evaluator reviewing a grading rubric and a student's answer.

    The current rubric is:
    {rubric}

    The student's answer is:
    {answer}

    Your task:
    - Suggest minimal modifications to the rubric to better align it with the student's answer.


    Format:
    <start>
    [Suggestion or 'No modification needed.']
    <end>
    """

    response = use_groq(textwrap.dedent(prompt).strip())
    return response.content

    # - Check if the answer contains significant correct concepts or reasoning steps that are not covered by any rubric point.
    # - If so, suggest a minimal modification (add, merge, or adjust points).
    # - Otherwise, explicitly say: "No modification needed."

question="write 10 lines on generative AI"
marks=10
demo_answer="""
Generative AI is a type of artificial intelligence technology capable of producing various types of content, including text, images, audio, and synthetic data.
Unlike traditional AI that primarily analyzes data, generative AI uses machine learning models to create novel outputs based on the patterns it learned from vast datasets.
Key technologies driving this field include large language models (LLMs) like GPT-4 and image generators like DALL-E and Stable Diffusion.
These models use deep learning techniques, particularly neural networks like transformers, to understand context and generate coherent and realistic outputs.
Generative AI has a wide range of applications across many industries, from automating content creation in media to accelerating drug discovery in healthcare.
It can assist in programming by writing code snippets, debugging, and explaining complex logic.
The technology can significantly boost productivity by automating repetitive tasks and providing creative assistance to professionals in design and marketing.
However, the use of generative AI also raises important ethical considerations, such as the potential for misuse in creating deepfakes and the implications for job displacement.
Data privacy concerns are also significant, as these models are trained on massive amounts of data, which may include personal information.
The field is rapidly evolving, with ongoing advancements aiming to improve the accuracy, efficiency, and ethical governance of these powerful tools.
"""

if __name__=="__main__":
    rubric=generate_rubric(question,marks,endpoint='groq')
    rubric=parse_rubric(rubric)
    print('generated rubric is:')
    for k,v in rubric.items():
        print(f'Rubric Point: {k}\nMarks: {v}\n')
    
    # raise Exception('Type(rubric)= '+str(type(rubric)))
    # answer_segments=break_answer_into_points(demo_answer,rubic,endpoint='groq')
    answer_segments=break_answer_into_points(demo_answer,rubric,endpoint='deberta')
    print('\n--- Answer Segments ---')
    segments=parse_answer_segments(answer_segments)
    for rubric_point, answer_part in segments.items():
        print(f'Rubric Point: {rubric_point}\nCorresponding Part: {answer_part}\n')