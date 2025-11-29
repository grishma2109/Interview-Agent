# interview_engine.py - uses Google Gemini (Google Gen AI SDK) with local fallback
import os
from typing import List, Tuple
import time

# Attempt to import Google GenAI SDK
try:
    # Official package: `google-genai`
    from google import genai
    _HAS_GENAI = True
except Exception:
    _HAS_GENAI = False

# If genai is available, create a client when needed
def _get_genai_client():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        # try environment or Streamlit secrets should set env var before calls
        raise RuntimeError("GOOGLE_API_KEY not set in environment.")
    # instantiate client
    try:
        client = genai.Client(api_key=key)
    except Exception as e:
        # Some older docs use genai.Client(); try both
        client = genai.Client()
        # rely on genai picking up environment or raise error later
    return client

# Helper to call Gemini for a text reply (simple wrapper)
def _genai_chat(prompt: str, model: str = "gemini-2.5-flash", max_output_tokens: int = 512) -> str:
    """
    Generate text via Google GenAI SDK (Gemini models).
    Uses client.models.generate_content(...) which returns an object with `.text` or `.content`.
    """
    if not _HAS_GENAI:
        raise RuntimeError("google-genai SDK not installed.")
    client = _get_genai_client()
    # Some versions expose client.models.generate_content
    try:
        response = client.models.generate_content(model=model, contents=prompt, max_output_tokens=max_output_tokens)
        # response may have .text or `.candidates`
        try:
            # Newer SDKs: response.text
            out = response.text
        except Exception:
            # Fallback: inspect candidates
            out = None
            if hasattr(response, "candidates") and response.candidates:
                out = response.candidates[0].content.get("text") if response.candidates[0].content else None
            if out is None:
                out = str(response)
        return out.strip()
    except Exception as e:
        # Re-raise to be handled upstream
        raise

# Local fallback generators / evaluators (used when API is unavailable)
def local_fallback_tech_questions(skills: List[str], role: str, count: int = 5) -> List[str]:
    qs = []
    if not skills:
        qs = [
            f"What motivated you to apply for the {role} role?",
            "Describe a meaningful project you worked on and your contribution.",
            "How do you approach debugging when you hit a tough bug?",
            "Explain a technical concept you are comfortable with to a non-technical person.",
            "How do you keep your technical skills up to date?"
        ]
    else:
        for i, s in enumerate(skills[:count]):
            qs.append(f"Describe a project where you used {s}. What challenges did you face and how did you solve them?")
        i = len(qs)
        while len(qs) < count:
            qs.append(f"Give a deep-dive explanation of {skills[i % len(skills)] if skills else 'a core technology you know'}.")
            i += 1
    return qs[:count]

def local_fallback_hr_questions(count: int = 4) -> List[str]:
    return [
        "Tell me about a time you faced a conflict at work and how you resolved it.",
        "What are your career goals for the next 2-3 years?",
        "How do you handle feedback and criticism?",
        "Why do you want to work at this company?"
    ]

def local_fallback_evaluator(question: str, answer: str, qtype: str, role: str) -> Tuple[float, str]:
    length_score = min(5, max(1, len(answer.split()) // 30))
    remarks = "Fallback heuristic evaluation: scored by answer length and skill presence."
    return float(length_score), remarks

# Public API functions used by the Streamlit app
def generate_technical_questions(resume_text: str, role: str, count: int = 5) -> List[str]:
    """
    Generate technical questions using Gemini. If Gemini is unavailable, return local fallback.
    """
    prompt = f"""You are an interview question generator for technical roles.
Given the role: "{role}" and the following resume content, produce {count} focused technical interview questions that test the candidate's skills and experience.
Return each question on its own line.

Resume:
{resume_text}
"""
    # Try Gemini
    if _HAS_GENAI:
        try:
            out = _genai_chat(prompt, model="gemini-2.5-flash", max_output_tokens=600)
            # split lines robustly
            lines = [ln.strip("-•0123456789. \t") for ln in out.splitlines() if ln.strip()]
            return lines[:count] if lines else local_fallback_tech_questions([], role, count=count)
        except Exception as e:
            # on any API error fallback
            return local_fallback_tech_questions([], role, count=count)
    else:
        return local_fallback_tech_questions([], role, count=count)

def generate_hr_questions(count: int = 4) -> List[str]:
    prompt = f"""You are an HR interviewer. Provide {count} common HR interview questions that assess communication, culture fit, career goals, and situational behavior. Return one question per line."""
    if _HAS_GENAI:
        try:
            out = _genai_chat(prompt, model="gemini-2.5-flash", max_output_tokens=300)
            lines = [ln.strip("-•0123456789. \t") for ln in out.splitlines() if ln.strip()]
            return lines[:count] if lines else local_fallback_hr_questions(count=count)
        except Exception:
            return local_fallback_hr_questions(count=count)
    else:
        return local_fallback_hr_questions(count=count)

def evaluate_answer_with_llm(question: str, answer_text: str, qtype: str, role: str) -> Tuple[float, str]:
    """
    Use Gemini to evaluate candidate answer. Returns (score, remarks).
    If Gemini fails or is unavailable, use a local heuristic.
    """
    eval_prompt = f"""
You are an interviewer evaluator. The role is "{role}".
Evaluate the candidate's answer to the question below.

Question:
{question}

Candidate answer:
{answer_text}

Instructions:
1) Provide a numeric score from 1 to 5 where 5 means excellent (deep technical understanding / clear examples).
2) Provide a short 1-2 sentence remark that highlights strengths/weaknesses and if follow-ups are needed.

Return the numeric score on the first line and the remarks on the next line only.
"""
    if _HAS_GENAI:
        try:
            out = _genai_chat(eval_prompt, model="gemini-2.5-flash", max_output_tokens=200)
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            if not lines:
                return local_fallback_evaluator(question, answer_text, qtype, role)
            # parse first numeric score
            try:
                first = lines[0]
                import re
                m = re.search(r"([1-5](?:\.\d)?)", first)
                if m:
                    score = float(m.group(1))
                else:
                    score = float(first.split()[0])
            except Exception:
                score = 3.0
            remarks = lines[1] if len(lines) > 1 else ""
            return float(score), remarks
        except Exception:
            return local_fallback_evaluator(question, answer_text, qtype, role)
    else:
        return local_fallback_evaluator(question, answer_text, qtype, role)
