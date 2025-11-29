# resume_parser.py
import re
from pathlib import Path
from typing import Tuple, List
from pypdf import PdfReader

# Simple skill list you can expand
COMMON_SKILLS = [
    "python","java","c++","c","javascript","react","node","django","flask",
    "sql","mysql","postgresql","mongodb","aws","azure","gcp","docker","kubernetes",
    "machine learning","deep learning","pytorch","tensorflow","nlp","computer vision",
    "html","css"
]

def extract_text_from_pdf(path:str)->str:
    path = Path(path)
    reader = PdfReader(str(path))
    texts = []
    for p in reader.pages:
        try:
            t = p.extract_text()
        except Exception:
            t = ""
        if t:
            texts.append(t)
    return "\n".join(texts)

def simple_skill_match(text:str, skills_list=COMMON_SKILLS)->List[str]:
    text_low = text.lower()
    found = set()
    for s in skills_list:
        if s in text_low:
            found.add(s)
    return sorted(list(found))

def extract_experience_years(text:str)->float:
    # very naive: look for patterns like "X years"
    matches = re.findall(r"(\d+)\+?\s+years", text.lower())
    if not matches:
        return 0.0
    nums = list(map(int, matches))
    return max(nums)

def extract_skills_and_summary(text:str)->Tuple[List[str], str]:
    skills = simple_skill_match(text)
    yrs = extract_experience_years(text)
    # make a short summary
    summary_lines = []
    if skills:
        summary_lines.append(f"Detected skills: {', '.join(skills[:10])}")
    if yrs:
        summary_lines.append(f"Experience (approx): {yrs} years")
    summary = " | ".join(summary_lines) if summary_lines else "No skills detected"
    return skills, summary
