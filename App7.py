# App5.py
# Streamlit app — hybrid of App2 + App4 with:
# - mixed grading panel (show extracted part + highlight)
# - tentative AI grading (prefill scores)
# - editable rubric post-generation
import streamlit as st
import re
import nltk
# make sure nltk punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

import hashlib
from Automations import (
    generate_rubric_2,
    break_answer_into_points,
    suggest_rubric_modification,
    ai_grade_segments,
)
from Parsers import parse_rubric, parse_answer_segments, parse_tentative_scores

# --- Page Config ---
st.set_page_config(
    page_title="AI-Assisted Grading Tool (App5)",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI-Assisted Grading Tool — App5 (Smart Grading Assistant)")
st.write(
    "Generates a rubric, extracts relevant answer parts, offers tentative AI grades, "
    "and lets you (the professor) finalize scores. Rubric is editable after generation."
)

# --- Helpers ---
def _safe_key(text: str):
    """Create a short stable key for session_state from arbitrary text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]

def highlight_sentence_wise(full_answer: str, extracted_segment: str):
    """
    Highlight each matching sentence from extracted_segment individually
    in the full answer.
    """

    # If nothing to highlight → return original wrapped in <pre>
    if not extracted_segment or extracted_segment.strip().lower() in {"not addressed", "not addressed."}:
        return f"<pre style='white-space: pre-wrap; font-family: inherit;'>{full_answer}</pre>"

    # Split extracted answer into sentences
    sentences = nltk.sent_tokenize(extracted_segment)

    highlighted = full_answer

    for sent in sentences:
        sent_clean = sent.strip()
        if len(sent_clean) < 3:
            continue  # ignore extremely tiny matches

        escaped = re.escape(sent_clean)

        # highlight only the first occurrence per sentence
        highlighted = re.sub(
            f"({escaped})",
            r"<span style='background-color: #fff176; color: black; padding: 2px 4px; border-radius: 4px;'>\1</span>",
            highlighted,
            count=1,
            flags=re.IGNORECASE
        )

    return f"<pre style='white-space: pre-wrap; font-family: inherit;'>{highlighted}</pre>"

# --- Session State Initialization ---
if 'rubric' not in st.session_state:
    st.session_state.rubric = None  # dict: {criterion_text: marks}
if 'raw_rubric_text' not in st.session_state:
    st.session_state.raw_rubric_text = None
if 'segments' not in st.session_state:
    st.session_state.segments = None  # OrderedDict from parse_answer_segments
if 'total_max_marks' not in st.session_state:
    st.session_state.total_max_marks = 0
if 'full_answer' not in st.session_state:
    st.session_state.full_answer = ""
if 'active_highlight' not in st.session_state:
    st.session_state.active_highlight = None
if 'ai_suggestions' not in st.session_state:
    st.session_state.ai_suggestions = {}
if 'use_ai_scores' not in st.session_state:
    st.session_state.use_ai_scores = True

# --- CSS Styling (from App4 + minor tweaks) ---
st.markdown(
    """
    <style>
    .answer-box {
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 8px;
        background-color: #ffffff;
        max-height: 400px;
        overflow-y: auto;
        /* white-space: pre-wrap; */ /* Handled by <pre> tag now */
        line-height: 1.6;
        color: black;
        font-size: 16px;
    }
    mark, .highlight {
        background-color: #ffeb3b !important;
        color: black !important;
        border-radius: 4px;
        padding: 2px 4px;
        animation: glow 1s ease-in-out;
    }
    @keyframes glow {
        0% { box-shadow: 0 0 5px #fff176; }
        50% { box-shadow: 0 0 15px #fdd835; }
        100% { box-shadow: 0 0 5px #fff176; }
    }
    .rubric-suggestion-box {
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #cccccc;
        border-radius: 10px;
        padding: 15px 18px;
        font-size: 16px;
        line-height: 1.6;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease-in-out;
    }
    .rubric-suggestion-header {
        font-weight: 600;
        font-size: 17px;
        margin-bottom: 6px;
        color: #1a1a1a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Step 1: Generate Rubric ---
st.header("1. Generate Rubric")
col1, col2 = st.columns([3, 1])

with col1:
    question = st.text_area("Enter the question:", height=100, placeholder="e.g., Explain the process of photosynthesis...")
    # --- MODIFICATION START ---
    # Added a new text area for demo answers
    demo_answers = st.text_area(
        "Enter demo/exemplar answers (optional):", 
        height=150, 
        placeholder="e.g., A perfect answer would mention...\nA poor answer might only..."
    )
    # --- MODIFICATION END ---
with col2:
    marks = st.number_input("Maximum Marks:", min_value=1, value=10)

if st.button("Generate Rubric", type="primary", use_container_width=True):
    if question and marks:
        try:
            with st.spinner("Generating rubric..."):
                # --- MODIFICATION START ---
                # Pass the new demo_answers text to the function
                raw_rubric = generate_rubric_2(question, marks, demo_answers, endpoint='groq')
                # --- MODIFICATION END ---
                parsed_rubric = parse_rubric(raw_rubric)
            if not parsed_rubric:
                st.error("AI failed to generate a rubric in the expected format. Check LLM output.")
            else:
                st.session_state.raw_rubric_text = raw_rubric
                st.session_state.rubric = parsed_rubric
                st.session_state.total_max_marks = sum(parsed_rubric.values())
                st.session_state.segments = None
                st.session_state.full_answer = ""
                st.session_state.ai_suggestions = {}
                st.success("Rubric generated successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)
    else:
        st.warning("Please enter both a question and max marks.")

# --- Rubric View + Editable UI ---
if st.session_state.rubric:
    st.divider()
    st.header("Generated Rubric (Editable)")

    with st.expander("View / Edit Generated Rubric", expanded=False):
        st.write("Current rubric points and marks (edit below, then click Save changes).")
        # We'll build editable fields indexed by position to avoid key collisions
        edited_items = []
        for idx, (criterion, m) in enumerate(list(st.session_state.rubric.items())):
            col_a, col_b, col_c = st.columns([6, 2, 1])
            with col_a:
                new_text = st.text_input(f"Criterion text #{idx+1}", value=criterion, key=f"edit_text_{idx}")
            with col_b:
                new_marks = st.number_input(f"Marks #{idx+1}", value=float(m), step=1.0, min_value=0.0, key=f"edit_marks_{idx}")
            with col_c:
                remove = st.checkbox("Remove", key=f"remove_{idx}")
            edited_items.append((new_text, int(new_marks), remove))

        if st.button("💾 Save Rubric Changes"):
            # Build new rubric dict (preserve order, skip removed)
            new_rubric = {}
            for text, mks, remove in edited_items:
                if remove:
                    continue
                cleaned = text.strip()
                if cleaned == "":
                    continue
                # if duplicate criterion texts, append index to make unique label
                if cleaned in new_rubric:
                    cleaned = f"{cleaned} (dup)"
                new_rubric[cleaned] = int(mks)
            if not new_rubric:
                st.warning("Rubric cannot be empty. No changes saved.")
            else:
                st.session_state.rubric = new_rubric
                st.session_state.total_max_marks = sum(new_rubric.values())
                st.success("Rubric updated.")
                # If we already have an answer, re-run segmentation and AI grading to align
                if st.session_state.full_answer:
                    try:
                        with st.spinner("Re-processing answer for updated rubric..."):
                            raw_segments = break_answer_into_points(
                                st.session_state.full_answer,
                                parse_rubric(st.session_state.raw_rubric_text) if st.session_state.raw_rubric_text else st.session_state.rubric,
                                endpoint=st.session_state.get("endpoint_choice", "groq"),
                            )
                            parsed_segments = parse_answer_segments(raw_segments)
                            st.session_state.segments = parsed_segments
                            # AI tentative grades
                            ai_out = ai_grade_segments(
                                st.session_state.full_answer,
                                st.session_state.rubric,
                                parsed_segments,
                                endpoint=st.session_state.get("endpoint_choice", "groq"),
                            )
                            # ai_grade_segments might return a dict or raw string; try to handle both
                            if isinstance(ai_out, dict):
                                st.session_state.ai_suggestions = ai_out
                            else:
                                try:
                                    st.session_state.ai_suggestions = parse_tentative_scores(ai_out)
                                except Exception:
                                    st.session_state.ai_suggestions = {}
                    except Exception as e:
                        st.error(f"Error re-processing after rubric edit: {e}")
                        st.exception(e)
                st.rerun()

# --- Step 2: Process Student's Answer ---
if st.session_state.rubric:
    st.divider()
    st.header("2. Process Student's Answer")

    with st.expander("View Generated Rubric (read-only)", expanded=False):
        st.write(st.session_state.rubric)

    # endpoint choice
    endpoint_choice = st.radio(
        "Select the model for answer processing:",
        ('groq', 'deberta', 'embedding_model'),
        index=0,
        horizontal=True,
        help="Groq is faster. DeBERTa might be better for QA-style extraction. embedding_model uses sentence embeddings."
    )
    st.session_state.endpoint_choice = endpoint_choice

    answer = st.text_area("Paste the student's answer here:", height=200, placeholder="The student's full answer...")
    use_ai = st.checkbox("Use AI's tentative marks as initial grades", value=True, key="use_ai_toggle")
    st.session_state.use_ai_scores = use_ai

    if st.button("Process Answer & Get AI Suggestions", type="primary", use_container_width=True):
        if not answer:
            st.warning("Please paste the student's answer.")
        else:
            try:
                st.session_state.full_answer = answer
                with st.spinner(f"Breaking down the answer using {endpoint_choice}..."):
                    raw_segments = break_answer_into_points(
                        answer,
                        parse_rubric(st.session_state.raw_rubric_text) if st.session_state.raw_rubric_text else st.session_state.rubric,
                        endpoint=endpoint_choice,
                    )
                    parsed_segments = parse_answer_segments(raw_segments)
                    st.session_state.segments = parsed_segments

                # Get tentative AI grades
                with st.spinner("Getting tentative AI grades..."):
                    ai_out = ai_grade_segments(
                        answer,
                        st.session_state.rubric,
                        parsed_segments,
                        endpoint=endpoint_choice,
                    )
                    if isinstance(ai_out, dict):
                        st.session_state.ai_suggestions = ai_out
                    else:
                        try:
                            st.session_state.ai_suggestions = parse_tentative_scores(ai_out)
                        except Exception:
                            st.session_state.ai_suggestions = {}
                st.success("Answer processed and AI suggestions ready.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)

# --- Step 3: Grade Breakdown (show extracted part + highlight + scores) ---
if st.session_state.segments:
    st.divider()
    st.header("3. Grade Breakdown (AI tentative grades + your final marks)")

    # Ensure full answer is present
    if not st.session_state.full_answer:
        st.error("Full answer text not found. Please paste and re-process the student's answer.")
        st.stop()

    full_answer_text = st.session_state.full_answer

    # Display full answer with possible highlight
    highlighted_answer = full_answer_text
    if st.session_state.active_highlight:
        seg_text = st.session_state.segments.get(st.session_state.active_highlight, "")
        highlighted_answer = highlight_sentence_wise(full_answer_text, seg_text)
        st.markdown(f"### Full Answer (Highlighting: *{st.session_state.active_highlight}*)")
    else:
        st.markdown("### Full Answer")
        # We still need to pass it through the function to get the <pre> tags
        highlighted_answer = highlight_sentence_wise(full_answer_text, "")

    st.markdown(f"<div class='answer-box'>{highlighted_answer}</div>", unsafe_allow_html=True)

    st.divider()

    # Grading panel
    total_score = 0.0
    max_marks_total = st.session_state.total_max_marks
    total_score_placeholder = st.empty()

    # To keep stable ordering, iterate over rubric items
    for idx, (rubric_point, max_score) in enumerate(st.session_state.rubric.items()):
        with st.container():
            col_left, col_right = st.columns([3, 1])
            with col_left:
                st.write(f"**Criterion:** {rubric_point}")
                segment_text = st.session_state.segments.get(rubric_point, "Not addressed.")
                # Show the AI-extracted part under the criterion
                st.info(f"**AI-extracted answer part:**\n\n{segment_text}")

                # Highlight button toggles this rubric's highlight in full answer
                if st.button("Highlight Relevant Part", key=f"highlight_btn_{_safe_key(rubric_point)}"):
                    # toggle the active highlight
                    st.session_state.active_highlight = (
                        None if st.session_state.active_highlight == rubric_point else rubric_point
                    )
                    st.rerun()

                # show AI suggestion caption
                ai_sugg_score = st.session_state.ai_suggestions.get(rubric_point, None)
                if ai_sugg_score is not None:
                    st.caption(f"🤖 AI Suggested (tentative): {ai_sugg_score} / {max_score}")

            with col_right:
                # prepare score key
                score_key = f"score_{_safe_key(rubric_point)}"
                # initialize session state for score if not present
                if score_key not in st.session_state:
                    # If use_ai_scores is True and ai has suggestion, prefill; else 0.0
                    init_val = float(ai_sugg_score) if (st.session_state.use_ai_scores and ai_sugg_score is not None) else 0.0
                    st.session_state[score_key] = init_val

                score = st.number_input(
                    f"Score (Max: {max_score})",
                    min_value=0.0,
                    max_value=float(max_score),
                    step=0.5,
                    value=float(st.session_state[score_key]),
                    key=score_key,
                )
                # persist selected score in session_state (number_input already does)
                # st.session_state[score_key] = score
                total_score += float(score)

    # show total
    total_score_placeholder.metric("TOTAL SCORE", f"{total_score} / {max_marks_total}", delta_color="off")

# --- Step 4: AI-Suggested Rubric Modification (from App4) ---
if 'rubric_suggestion' not in st.session_state:
    st.session_state.rubric_suggestion = None

# If we have full answer & rubric, compute suggestion (non-blocking—user triggers)
if st.session_state.rubric and st.session_state.full_answer:
    st.divider()
    st.header("4. AI-Suggested Rubric Modification")
    if st.button("Analyze for rubric modification", key="analyze_rubric_mod"):
        try:
            with st.spinner("Analyzing if rubric modifications are needed..."):
                suggestion_text = suggest_rubric_modification(
                    st.session_state.full_answer,
                    st.session_state.rubric,
                    endpoint='groq'
                )
                if "<start>" in suggestion_text and "<end>" in suggestion_text:
                    suggestion = suggestion_text.split("<start>")[1].split("<end>")[0].strip()
                else:
                    suggestion = suggestion_text.strip()
                st.session_state.rubric_suggestion = suggestion
        except Exception as e:
            st.error(f"Error while analyzing: {e}")
            st.exception(e)

    if st.session_state.rubric_suggestion:
        suggestion = st.session_state.rubric_suggestion
        if suggestion.lower().startswith("no modification"):
            st.success("✅ No rubric modification needed. The rubric fits the student's answer well.")
        else:
            with st.expander("💡 Suggested Rubric Adjustment", expanded=True):
                st.markdown(
                    f"""
                    <div class="rubric-suggestion-box">
                        <div class="rubric-suggestion-header">AI Suggestion:</div>
                        {suggestion}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if st.button("Want to Add Suggested Point?", key="add_suggestion_btn"):
                # naive: add first line as new point with 1 mark (prof may edit later)
                new_point = suggestion.split('\n')[0].strip()
                if new_point:
                    # avoid duplicates
                    if new_point in st.session_state.rubric:
                        st.warning("Rubric already contains a similar point.")
                    else:
                        st.session_state.rubric[new_point] = 1
                        st.session_state.total_max_marks = sum(st.session_state.rubric.values())
                        st.success("Suggested rubric modification added. Please review/edit if needed.")
                        st.rerun()

# --- Footer / Save / Export suggestion ---
st.divider()
st.caption("App5: Mixed view (AI + Professor). AI gives suggestions; professor gives final grades. Rubric is editable. ✅")