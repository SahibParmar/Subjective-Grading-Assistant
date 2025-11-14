import os
import re
from collections import OrderedDict

def parse_answer_segments(response_content):
    rubric_dict = OrderedDict()

    # Find the content between <start> and <end>
    start_index = response_content.find('<start>')
    end_index = response_content.find('<end>')

    if start_index != -1 and end_index != -1:
        content = response_content[start_index + len('<start>'):end_index].strip()
        # Split the content by '####' to get individual rubric points
        segments = [seg.strip() for seg in content.split('####') if seg.strip()]

        for point in segments:
            rubric_match = re.search(r'Rubric:\s*(.*)', point)
            part_match = re.search(r'corresponding_part:\s*(.*)', point)

            if rubric_match and part_match:
                rubric_text = rubric_match.group(1).strip()
                part_text = part_match.group(1).strip()
                rubric_dict[rubric_text] = part_text
            elif rubric_match:
                print(f"⚠️ Warning: Could not find corresponding part for rubric '{rubric_match.group(1).strip()}'")
            elif part_match:
                print(f"⚠️ Warning: Could not find rubric for part '{part_match.group(1).strip()}'")
    else:
        print("❌ Could not find <start> or <end> tags in the response.")

    return rubric_dict



def parse_rubric(response_content):
    rubric_dict = {}
    # Find the content between <start> and <end>
    start_index = response_content.find('<start>')
    end_index = response_content.find('<end>')

    if start_index != -1 and end_index != -1:
        content = response_content[start_index + len('<start>'):end_index].strip()
        # Split the content by '####' to get individual rubric points
        rubric_points = content.split('####')

        for point in rubric_points:
            # Extract Rubric and Marks from each point
            rubric_match = re.search(r'Rubric:(.*?)\n', point)
            marks_match = re.search(r'Marks:(.*?)\n', point)

            if rubric_match and marks_match:
                rubric_text = rubric_match.group(1).strip()
                marks_text = marks_match.group(1).strip()
                try:
                    marks_value = int(marks_text)
                    rubric_dict[rubric_text] = marks_value
                except ValueError:
                    print(f"Warning: Could not convert marks '{marks_text}' to integer for rubric '{rubric_text}'")
            elif rubric_match:
                 print(f"Warning: Could not find marks for rubric '{rubric_match.group(1).strip()}'")
            elif marks_match:
                 print(f"Warning: Could not find rubric for marks '{marks_match.group(1).strip()}'")

    else:
        print("Could not find <start> or <end> tags in the response.")

    return rubric_dict

def parse_tentative_scores(response_content):
    scores = {}
    start_index = response_content.find('<start>')
    end_index = response_content.find('<end>')
    if start_index != -1 and end_index != -1:
        body = response_content[start_index+len('<start>'):end_index].strip()
        for seg in body.split('####'):
            rubric_match = re.search(r'Rubric:\s*(.*)', seg)
            score_match = re.search(r'Tentative_Score:\s*(.*)', seg)
            if rubric_match and score_match:
                scores[rubric_match.group(1).strip()] = float(score_match.group(1).strip())
    return scores
