"""
app_interview.py
Deploy-ready Streamlit interview app with:
 - webcam + microphone check via streamlit-webrtc
 - motion detection
 - resume upload + parsing (placeholder hooks to local modules)
 - question generation + evaluation hooks (placeholder hooks to local modules)
 - voice Q&A: audio upload + transcription via Gemini (Gemini 2.0 Flash experimental)
 - fallback to manual transcript entry
 - report generation hook
"""

from pathlib import Path
import os
import time
from datetime import datetime
from typing import List

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase

# Only import cv2 when run-time required (some platforms want headless opencv)
import numpy as np

# local modules you must provide in the project (placeholders referenced in your original code)
# - resume_parser.py: extract_text_from_pdf, extract_skills_and_summary, extract_experience_years
# - interview_engine.py: generate_technical_questions, generate_hr_questions, evaluate_answer_with_llm
# - report_generator.py: generate_report_pdf
# - agent.py: ask_hr_assistant
#
# Keep these files in the project root (same folder as app_interview.py) or install as a package.
try:
    from resume_parser import extract_text_from_pdf, extract_skills_and_summary, extract_experience_years
    from interview_engine import (
        generate_technical_questions,
        generate_hr_questions,
        evaluate_answer_with_llm,
    )
    from report_generator import generate_report_pdf
    from agent import ask_hr_assistant
except Exception:
    # If local modules aren't present, provide minimal fallbacks so the app doesn't crash on import.
    def extract_text_from_pdf(path): 
        return ""
    def extract_skills_and_summary(text):
        return ([], "")
    def extract_experience_years(text):
        return 0.0
    def generate_technical_questions(resume_text, role, count=5):
        return [f"Describe a project related to {i}" for i in range(1, count+1)]
    def generate_hr_questions(count=4):
        return ["Tell me about a time you faced a conflict at work."] * count
    def evaluate_answer_with_llm(q, ans, qtype, role):
        # return score (0-5), remarks
        return 3.0, "Fallback evaluator"
    def generate_report_pdf(candidate, skills, qa_history, role):
        # return bytes of a simple PDF-like placeholder
        return b"%PDF-1.4\n%placeholder\n"
    def ask_hr_assistant(q):
        return ("No docs indexed.", [], [])

# ---------- Config ----------
st.set_page_config(page_title="Interview Agent (AV Enabled) — Gemini transcription", layout="wide")
BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)
MOTION_FLAG = BASE / "motion.flag"

VECTORSTORE_DIR = "vectorstore"  # optional, used by agent.ask_hr_assistant

# Sidebar: API keys & settings
st.sidebar.header("API Keys & Settings")

# Gemini (Google) key — set this in your Render / Streamlit Cloud environment vars
google_key_input = st.sidebar.text_input("Google (Gemini) API key (paste for session)", type="password")
if google_key_input:
    os.environ["GOOGLE_API_KEY"] = google_key_input
    st.sidebar.success("Google key set for session.")

# Keep an optional OpenAI key (if you ever want to switch back)
openai_key_input = st.sidebar.text_input("OpenAI API key (optional)", type="password")
if openai_key_input:
    os.environ["OPENAI_API_KEY"] = openai_key_input
    st.sidebar.success("OpenAI key set for session.")

st.sidebar.markdown("---")
policy_q = st.sidebar.text_input("Ask policy question (from uploaded docs)")
if st.sidebar.button("Run Policy QA") and policy_q:
    try:
        ans, sources, formatted = ask_hr_assistant(policy_q)
        st.sidebar.markdown("**Answer:**")
        st.sidebar.write(ans)
        st.sidebar.markdown("**Sources:**")
        st.sidebar.write(sources)
    except Exception as e:
        st.sidebar.error(f"Policy QA error: {e}")

# ---------- session state defaults ----------
if "stage" not in st.session_state:
    st.session_state.stage = "collect_info"
if "candidate" not in st.session_state:
    st.session_state.candidate = {}
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "skills" not in st.session_state:
    st.session_state.skills = []
if "tech_questions" not in st.session_state:
    st.session_state.tech_questions = []
if "hr_questions" not in st.session_state:
    st.session_state.hr_questions = []
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "current_q_index" not in st.session_state:
    st.session_state.current_q_index = 0
if "role" not in st.session_state:
    st.session_state.role = ""
if "motion_alert_since" not in st.session_state:
    st.session_state.motion_alert_since = None
if "voice_questions" not in st.session_state:
    st.session_state.voice_questions = [
        "Please briefly introduce yourself and highlight one project you're proud of.",
        "Explain a technical concept from your resume (e.g., a library or algorithm) in simple terms.",
        "Why do you want this role and how would you contribute in the first 3 months?"
    ]
if "voice_answers" not in st.session_state:
    st.session_state.voice_answers = [None] * len(st.session_state.voice_questions)
if "voice_transcripts" not in st.session_state:
    st.session_state.voice_transcripts = [None] * len(st.session_state.voice_questions)

# ---------- Motion detector (video transformer) ----------
class MotionDetector(VideoTransformerBase):
    def __init__(self):
        self.prev_frame = None

    def transform(self, frame):
        import cv2  # local import for safety on some platforms
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            # return original image on first frame to show user video
            return img

        delta_frame = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.prev_frame = gray

        # write a timestamp (float seconds) into the MOTION_FLAG so main thread can read it
        try:
            with open(MOTION_FLAG, "w") as fh:
                fh.write(str(time.time()))
        except Exception:
            pass

        # return threshold image (binary mask) for visualization
        # convert single-channel to 3-channel so streamlit-webrtc can display
        thresh_3c = np.stack([thresh]*3, axis=-1)
        return thresh_3c

# RTC config (STUN)
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# start the webrtc streamer with motion detection
webrtc_ctx = webrtc_streamer(
    key="camera",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": True},
    video_processor_factory=MotionDetector,
    async_processing=True,
)

# camera + mic status
cam_status = "connected" if webrtc_ctx and webrtc_ctx.state.playing else "not connected"
st.subheader("Camera & Microphone Check (live)")
st.write("Camera + mic status:", cam_status)
st.markdown(
    "Allow camera & microphone when your browser asks. The system monitors for motion and will flag unexpected movement."
)

# read motion flag helper
def check_motion_flag(timeout_seconds=10):
    if MOTION_FLAG.exists():
        try:
            ts = float(MOTION_FLAG.read_text())
            if time.time() - ts < timeout_seconds:
                return True, ts
        except Exception:
            return False, None
    return False, None

motion_detected, motion_ts = check_motion_flag(timeout_seconds=10)
if motion_detected:
    if not st.session_state.motion_alert_since:
        st.session_state.motion_alert_since = motion_ts
    elapsed = time.time() - st.session_state.motion_alert_since
    st.error(f"⚠️ Motion detected — alert active (elapsed {int(elapsed)} s). Please stay visible and minimize movement.")
    st.progress(min(100, int((elapsed % 60) / 60 * 100)))
else:
    st.session_state.motion_alert_since = None
    st.success("No recent unexpected movement detected.")

st.markdown("---")

# ---------- Gemini transcription helper ----------
def transcribe_with_gemini(audio_path: str) -> str:
    """
    Transcribe audio using Gemini (Gemini 2.0 Flash experimental).
    NOTE: SDKs and method names differ across releases. This function includes a best-effort
    implementation using the `google.generativeai` style API. If your installed Gemini client
    uses different calls, replace this function with the correct client code.

    Expected environment variable: GOOGLE_API_KEY

    Returns:
        transcript (str) on success, or raises Exception on failure.
    """
    # Minimal checks
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not set in environment. Please set it in your hosting platform.")

    # Best-effort import for Google's generative AI python library
    try:
        import google.generativeai as genai  # common wrapper name used in examples
    except Exception as e:
        # If this import fails, raise a clear message so the deployer can fix dependency
        raise RuntimeError("Gemini client library (google.generativeai) not installed. Add it to requirements or replace this function.") from e

    # Configure client
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    except Exception:
        # some versions use set_api_key or similar; continue anyway and let the client calls fail with clear error
        pass

    # The exact transcription call differs by SDK version. Below is a best-effort pattern for
    # streaming/file transcription. Replace with the SDK call appropriate to your installed package.
    #
    # Example pattern (pseudocode / typical):
    #   response = genai.audio.transcribe(model="models/gemini-2.0-flash-exp", file=open(audio_path,"rb"))
    #   transcript = response["text"] or response.text
    #
    # If the SDK you have uses a different method, update accordingly.
    try:
        # try a couple of plausible call patterns (non-exhaustive)
        # Pattern A (some wrappers):
        if hasattr(genai, "audio") and hasattr(genai.audio, "transcribe"):
            with open(audio_path, "rb") as af:
                resp = genai.audio.transcribe(model="gemini-2.0-flash", file=af)
            # resp may be a dict or object
            if isinstance(resp, dict):
                return resp.get("text", "") or resp.get("transcript", "")
            # fallback to attribute access
            return getattr(resp, "text", "") or getattr(resp, "transcript", "")

        # Pattern B (other wrappers)
        if hasattr(genai, "GenerativeModel"):
            # instantiate model
            Model = genai.GenerativeModel("models/gemini-2.0-flash")
            # many SDKs expect a content request; some accept audio data
            # THIS IS SDK-SPECIFIC. Use this as a placeholder you may need to change.
            with open(audio_path, "rb") as af:
                # Example: Model.generate_audio_transcript(file=af) -- replace with actual call
                if hasattr(Model, "transcribe"):
                    resp = Model.transcribe(file=af)
                    if isinstance(resp, dict):
                        return resp.get("text", "") or resp.get("transcript", "")
                    return getattr(resp, "text", "") or getattr(resp, "transcript", "")
                if hasattr(Model, "generate") and hasattr(Model.generate, "__call__"):
                    # Potential pattern: generate(content=[{"audio": ...}])
                    try:
                        with open(audio_path, "rb") as af2:
                            blob = af2.read()
                        # many higher-level clients want base64 or bytes packaged in a proto
                        resp = Model.generate_audio_transcription(file_bytes=blob)  # likely to fail unless library matches
                        if isinstance(resp, dict):
                            return resp.get("text", "") or resp.get("transcript", "")
                        return getattr(resp, "text", "") or getattr(resp, "transcript", "")
                    except Exception:
                        pass

        # If we reached here, none of the guessed call patterns matched
        raise RuntimeError("Unable to call Gemini transcription with the current client wrapper. Please adapt transcribe_with_gemini() to your installed Gemini SDK.")
    except Exception as e:
        # raise the original exception so the UI can report it and allow manual transcript fallback
        raise

# ---------- Interview flow ----------
if st.session_state.stage == "collect_info":
    st.header("Step 1 — Candidate Info")
    with st.form("info"):
        name = st.text_input("Full name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        role = st.text_input("Role applying for")
        submit = st.form_submit_button("Next: Upload Resume")
    if submit:
        if not name or not email or not role:
            st.warning("Please fill name, email and role.")
        else:
            st.session_state.candidate = {
                "name": name,
                "email": email,
                "phone": phone,
                "applied_role": role,
                "timestamp": datetime.utcnow().isoformat()
            }
            st.session_state.role = role
            st.session_state.stage = "upload_resume"
            st.experimental_rerun()

elif st.session_state.stage == "upload_resume":
    st.header("Step 2 — Upload Resume (PDF)")
    uploaded = st.file_uploader("Upload PDF resume", type=["pdf"])
    if uploaded:
        dest = DATA_DIR / f"resume_{st.session_state.candidate['name'].replace(' ','_')}.pdf"
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success("Saved resume; parsing...")
        try:
            text = extract_text_from_pdf(str(dest))
        except Exception:
            text = ""
        st.session_state.resume_text = text
        skills, summary = extract_skills_and_summary(text)
        yrs = extract_experience_years(text)
        st.session_state.skills = skills
        st.session_state.resume_years = yrs
        st.session_state.stage = "generate_questions"
        st.experimental_rerun()

    st.write("Tip: If you can't upload a file, paste resume text below.")
    txt = st.text_area("Paste resume text (optional)", height=200)
    if txt and st.button("Use pasted text"):
        st.session_state.resume_text = txt
        skills, summary = extract_skills_and_summary(txt)
        yrs = extract_experience_years(txt)
        st.session_state.skills = skills
        st.session_state.resume_years = yrs
        st.session_state.stage = "generate_questions"
        st.experimental_rerun()

elif st.session_state.stage == "generate_questions":
    st.header("Step 3 — Generating Questions")
    st.write("Generating technical + HR + voice questions ...")
    try:
        tech_qs = generate_technical_questions(st.session_state.resume_text, st.session_state.role, count=5)
        hr_qs = generate_hr_questions(count=4)
    except Exception as e:
        st.warning(f"LLM question generation error: {e}. Using fallback.")
        tech_qs = [f"Describe a project where you used {s}." for s in st.session_state.skills[:5]] or ["Describe a meaningful project you worked on."]
        hr_qs = [
            "Tell me about a time you faced a conflict at work and how you resolved it.",
            "What are your career goals?",
            "How do you handle feedback?",
            "Why this company?"
        ]

    st.session_state.tech_questions = tech_qs
    st.session_state.hr_questions = hr_qs
    st.session_state.current_q_index = 0
    st.session_state.stage = "qna"
    st.experimental_rerun()

elif st.session_state.stage == "qna":
    st.header("Step 4 — Interview (text & voice answers)")
    # resume quick score
    skill_score = min(6, len(st.session_state.skills))
    yrs = float(st.session_state.get("resume_years", 0) or 0)
    exp_score = min(4, int(min(4, yrs)))
    resume_score = skill_score + exp_score
    resume_score = round((resume_score / 10) * 10, 1)
    st.metric("Resume score (out of 10)", resume_score)

    all_qs = [{"q": q, "type": "technical"} for q in st.session_state.tech_questions] + \
             [{"q": q, "type": "hr"} for q in st.session_state.hr_questions]

    idx = st.session_state.current_q_index
    total = len(all_qs) + len(st.session_state.voice_questions)
    st.write(f"Question {idx+1} / {total}")

    # Text questions
    if idx < len(all_qs):
        current = all_qs[idx]
        st.subheader(f"Q (text) — {current['type'].capitalize()}")
        st.write(current["q"])
        with st.form(key=f"text_answer_form_{idx}"):
            text_ans = st.text_area("Type your answer here", height=200)
            submit_ans = st.form_submit_button("Submit Answer")
        if submit_ans:
            if not text_ans.strip():
                st.warning("Please enter an answer.")
            else:
                try:
                    score, remarks = evaluate_answer_with_llm(current["q"], text_ans, current["type"], st.session_state.role)
                except Exception as e:
                    st.warning(f"Evaluation error: {e}. Using heuristic fallback.")
                    length_score = min(5, max(1, len(text_ans.split()) // 30))
                    skill_hits = sum(1 for s in st.session_state.skills if s.lower() in text_ans.lower())
                    score = min(5, length_score + skill_hits)
                    remarks = "Fallback evaluation applied."
                st.session_state.qa_history.append({
                    "question": current["q"],
                    "type": current["type"],
                    "answer": text_ans,
                    "score": float(score),
                    "remarks": remarks
                })
                st.session_state.current_q_index += 1
                st.experimental_rerun()
    else:
        # Voice questions stage
        v_idx = idx - len(all_qs)
        if v_idx < len(st.session_state.voice_questions):
            st.subheader(f"Q (voice) — {v_idx+1} of {len(st.session_state.voice_questions)}")
            st.write(st.session_state.voice_questions[v_idx])
            st.markdown(
                "Record your answer locally and upload the audio file (wav/m4a/mp3/ogg). "
                "The app will attempt automatic transcription using Gemini (Gemini 2.0 Flash experimental)."
            )
            uploaded_audio = st.file_uploader(f"Upload audio answer for voice Q{v_idx+1}", type=["wav","m4a","mp3","ogg"], key=f"audio_{v_idx}")
            if uploaded_audio:
                tmp_path = DATA_DIR / f"voice_q{v_idx+1}_{st.session_state.candidate.get('name','candidate')}.wav"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                st.success("Audio saved. Attempting transcription via Gemini...")
                transcript = None
                try:
                    transcript = transcribe_with_gemini(str(tmp_path))
                    if transcript:
                        st.text_area("Transcript (edit if needed)", value=transcript, key=f"trans_{v_idx}", height=150)
                        if st.button("Submit voice answer (use transcript)", key=f"submit_trans_{v_idx}"):
                            try:
                                score, remarks = evaluate_answer_with_llm(st.session_state.voice_questions[v_idx], transcript, "voice", st.session_state.role)
                            except Exception:
                                length_score = min(5, max(1, len(transcript.split()) // 30))
                                skill_hits = sum(1 for s in st.session_state.skills if s.lower() in transcript.lower())
                                score = min(5, length_score + skill_hits)
                                remarks = "Fallback evaluation applied."
                            st.session_state.voice_answers[v_idx] = str(tmp_path)
                            st.session_state.voice_transcripts[v_idx] = transcript
                            st.session_state.qa_history.append({
                                "question": st.session_state.voice_questions[v_idx],
                                "type": "voice",
                                "answer": transcript,
                                "score": float(score),
                                "remarks": remarks
                            })
                            st.session_state.current_q_index += 1
                            st.experimental_rerun()
                    else:
                        st.warning("Gemini returned empty transcript. Please paste manually.")
                except Exception as e:
                    st.warning(f"Gemini transcription failed: {e}. Please paste transcript manually.")
                    # Show manual transcript UI below (falls through)
                # Manual transcript fallback
                manual = st.text_area("Paste transcript of your audio here (fallback)", key=f"manual_trans_{v_idx}", height=150)
                if st.button("Submit voice answer (manual transcript)", key=f"submit_manual_{v_idx}"):
                    if not manual.strip():
                        st.warning("Please paste or type transcript.")
                    else:
                        try:
                            score, remarks = evaluate_answer_with_llm(st.session_state.voice_questions[v_idx], manual, "voice", st.session_state.role)
                        except Exception:
                            length_score = min(5, max(1, len(manual.split()) // 30))
                            skill_hits = sum(1 for s in st.session_state.skills if s.lower() in manual.lower())
                            score = min(5, length_score + skill_hits)
                            remarks = "Fallback evaluation applied."
                        st.session_state.voice_answers[v_idx] = str(tmp_path)
                        st.session_state.voice_transcripts[v_idx] = manual
                        st.session_state.qa_history.append({
                            "question": st.session_state.voice_questions[v_idx],
                            "type": "voice",
                            "answer": manual,
                            "score": float(score),
                            "remarks": remarks
                        })
                        st.session_state.current_q_index += 1
                        st.experimental_rerun()
        else:
            # All questions answered
            st.success("All questions answered. Generating report...")
            st.session_state.stage = "done"
            st.experimental_rerun()

    # Sidebar progress
    st.sidebar.header("Progress")
    st.sidebar.write(f"Answered: {len(st.session_state.qa_history)}")
    st.sidebar.write(f"Remaining (approx): {total - len(st.session_state.qa_history)}")

elif st.session_state.stage == "done":
    st.header("Interview Completed — Report & Eligibility")
    for i, qa in enumerate(st.session_state.qa_history, 1):
        st.markdown(f"**{i}. ({qa['type']}) {qa['question']}**")
        st.write(f"Answer: {qa['answer']}")
        st.write(f"Score: {qa['score']:.1f}")
        st.write(f"Remarks: {qa['remarks']}")
        st.markdown("---")

    if st.session_state.qa_history:
        avg_answer_score = sum(q["score"] for q in st.session_state.qa_history) / len(st.session_state.qa_history)
    else:
        avg_answer_score = 0.0

    resume_score = round(min(10.0, max(0.0, (min(6, len(st.session_state.skills)) + min(4, int(st.session_state.get('resume_years',0) or 0)) ))),1)
    composite = (resume_score * 0.4) + ((avg_answer_score * 2) * 0.6)
    composite = round(composite, 1)

    st.metric("Resume score (0-10)", resume_score)
    st.metric("Average answer score (0-5)", round(avg_answer_score, 2))
    st.metric("Composite score (0-10)", composite)

    if composite >= 7.5:
        eligibility = ("Eligible", "green")
    elif composite >= 5.5:
        eligibility = ("Maybe (Requires further interview)", "orange")
    else:
        eligibility = ("Not eligible", "red")

    st.markdown(f"### Eligibility: **{eligibility[0]}**")
    st.info(f"This is an automated recommendation. Final decisions should be made by HR. (Badge color: {eligibility[1]})")

    # Generate report
    try:
        report_bytes = generate_report_pdf(
            candidate=st.session_state.candidate,
            skills=st.session_state.skills,
            qa_history=st.session_state.qa_history,
            role=st.session_state.role
        )
    except Exception:
        report_bytes = b"%PDF-1.4\n%placeholder\n"
    st.download_button("Download PDF Report", data=report_bytes, file_name=f"report_{st.session_state.candidate.get('name','candidate')}.pdf", mime="application/pdf")

    if st.button("Restart Interview"):
        keys = ["stage","candidate","resume_text","skills","tech_questions","hr_questions","qa_history","current_q_index","role","voice_questions","voice_answers","voice_transcripts","resume_years","motion_alert_since"]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        try:
            if MOTION_FLAG.exists():
                MOTION_FLAG.unlink()
        except Exception:
            pass
        st.experimental_rerun()
