# app_interview.py (patched — safe fallback for cv2 & streamlit-webrtc)
import os
import io
import time
from pathlib import Path
from datetime import datetime
from typing import List
import pickle

import streamlit as st
import numpy as np

# local modules (these must exist in your project)
# resume_parser, interview_engine, report_generator, agent
from resume_parser import extract_text_from_pdf, extract_skills_and_summary, extract_experience_years
from interview_engine import (
    generate_technical_questions,
    generate_hr_questions,
    evaluate_answer_with_llm,
)
from report_generator import generate_report_pdf
from agent import ask_hr_assistant

# --- Config & paths ---
st.set_page_config(page_title="Interview Agent (AV Enabled)", layout="wide")
BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

VECTORSTORE_DIR = "vectorstore"  # if you use policy QA

# small helper file used to signal motion (bridge between video thread and main thread)
MOTION_FLAG = BASE / "motion.flag"

st.title("🤖 Interview Agent — with AV monitoring, voice Q&A, resume rating")

# ---------- Sidebar: keys & status ----------
st.sidebar.header("API Keys & Settings")
# Load keys from st.secrets if present
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

google_key_input = st.sidebar.text_input("Google (Gemini) API key (paste for session)", type="password")
if google_key_input:
    os.environ["GOOGLE_API_KEY"] = google_key_input
    st.sidebar.success("Google key set for session.")

openai_key_input = st.sidebar.text_input("OpenAI API key (paste for session, optional for Whisper)", type="password")
if openai_key_input:
    os.environ["OPENAI_API_KEY"] = openai_key_input
    st.sidebar.success("OpenAI key set for session.")

st.sidebar.markdown("---")
policy_q = st.sidebar.text_input("Ask policy question (from uploaded docs)")
if st.sidebar.button("Run Policy QA") and policy_q:
    try:
        ans, sources = ask_hr_assistant(policy_q)
        st.sidebar.markdown("**Answer:**")
        st.sidebar.write(ans)
        st.sidebar.markdown("**Sources:**")
        st.sidebar.write(sources)
    except Exception as e:
        st.sidebar.error(f"Policy QA error: {e}")

# ---------- session state ----------
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

# ---------- Safe imports for cv2 and streamlit-webrtc ----------
# cv2 (OpenCV)
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False

# streamlit-webrtc
WEBCAM_AVAILABLE = False
try:
    from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBCAM_AVAILABLE = True
except Exception:
    VideoTransformerBase = object

    class _DummyWebRtcMode:
        SENDRECV = "sendrecv"
    WebRtcMode = _DummyWebRtcMode

    class _DummyRTCConfiguration:
        def __init__(self, config):
            self.configuration = config
    RTCConfiguration = _DummyRTCConfiguration

    class _DummyState:
        def __init__(self):
            self.playing = False

    class _DummyCtx:
        def __init__(self):
            self.state = _DummyState()

    def webrtc_streamer(*args, **kwargs):
        st.warning("Video features disabled (streamlit-webrtc not available on this host).")
        return _DummyCtx()

# Inform user if motion detection will be disabled
if not CV2_AVAILABLE:
    st.sidebar.warning("OpenCV (`cv2`) not installed on this host — motion detection disabled. Install `opencv-python-headless` locally to enable it.")

# ---------- Webcam + motion detection UI ----------
st.subheader("Camera & Microphone Check (live)")
st.markdown(
    "Allow camera & microphone when your browser asks. "
    "The system will monitor for unexpected movements and show an alert when detected (if supported)."
)

# Video transformer using optional OpenCV for motion detection
class MotionDetector(VideoTransformerBase):
    def __init__(self):
        self.prev_frame = None

    def transform(self, frame):
        # If cv2 not available, fallback to returning raw frame
        if not CV2_AVAILABLE:
            try:
                img = frame.to_ndarray(format="bgr24")
                return img
            except Exception:
                return frame

        # Normal path with OpenCV available
        try:
            img = frame.to_ndarray(format="bgr24")
        except Exception:
            img = np.asarray(frame)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return img

        delta_frame = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        self.prev_frame = gray
        try:
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        except Exception:
            return thresh

# RTC configuration (if available)
RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
rtc_conf_obj = RTCConfiguration(RTC_CONFIG) if WEBCAM_AVAILABLE else None

# Start or fallback webrtc streamer
if WEBCAM_AVAILABLE:
    webrtc_ctx = webrtc_streamer(
        key="camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_conf_obj,
        media_stream_constraints={"video": True, "audio": True},
        video_processor_factory=MotionDetector,
    )
else:
    webrtc_ctx = webrtc_streamer(key="camera")  # returns dummy ctx if not available

# Show mic/cam status safely
cam_playing = False
try:
    cam_playing = bool(getattr(getattr(webrtc_ctx, "state", None), "playing", False))
except Exception:
    cam_playing = False
cam_status = "connected" if cam_playing else "not connected"
st.write("Camera + mic status:", cam_status)

# motion alert check: read the flag file's timestamp within last N seconds
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

# ---------- Interview flow (basic info, resume upload) ----------
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
            st.session_state.candidate = {"name": name, "email": email, "phone": phone, "applied_role": role, "timestamp": datetime.utcnow().isoformat()}
            st.session_state.role = role
            st.session_state.stage = "upload_resume"
            st.rerun()

elif st.session_state.stage == "upload_resume":
    st.header("Step 2 — Upload Resume (PDF)")
    uploaded = st.file_uploader("Upload PDF resume", type=["pdf"])
    if uploaded:
        dest = DATA_DIR / f"resume_{st.session_state.candidate['name'].replace(' ','_')}.pdf"
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success("Saved resume; parsing...")
        text = extract_text_from_pdf(str(dest))
        st.session_state.resume_text = text
        skills, summary = extract_skills_and_summary(text)
        yrs = extract_experience_years(text)
        st.session_state.skills = skills
        st.session_state.resume_years = yrs
        st.session_state.stage = "generate_questions"
        st.rerun()
    st.write("Tip: If you can't upload a file, paste resume text below.")
    txt = st.text_area("Paste resume text (optional)", height=200)
    if txt and st.button("Use pasted text"):
        st.session_state.resume_text = txt
        skills, summary = extract_skills_and_summary(txt)
        yrs = extract_experience_years(txt)
        st.session_state.skills = skills
        st.session_state.resume_years = yrs
        st.session_state.stage = "generate_questions"
        st.rerun()

# generate questions
elif st.session_state.stage == "generate_questions":
    st.header("Step 3 — Generating Questions")
    st.write("Generating technical + HR + voice questions (2-3) ...")
    try:
        tech_qs = generate_technical_questions(st.session_state.resume_text, st.session_state.role, count=5)
        hr_qs = generate_hr_questions(count=4)
    except Exception as e:
        st.warning(f"LLM question generation error: {e}. Using local fallback.")
        tech_qs = [
            f"Describe a project where you used {s}." for s in st.session_state.skills[:5]
        ] or ["Describe a meaningful project you worked on."]
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
    st.rerun()

# interactive Q&A + voice answers
elif st.session_state.stage == "qna":
    st.header("Step 4 — Interview (text & voice answers)")

    # Show resume rating quick summary
    skill_score = min(6, len(st.session_state.skills))  # up to 6 points
    yrs = float(st.session_state.get("resume_years", 0) or 0)
    exp_score = min(4, int(min(4, yrs)))  # up to 4 points
    resume_score = skill_score + exp_score
    resume_score = round((resume_score / 10) * 10, 1)  # already out of 10
    st.metric("Resume score (out of 10)", resume_score)

    # Combine questions: technical then HR
    all_qs = [{"q": q, "type": "technical"} for q in st.session_state.tech_questions] + \
             [{"q": q, "type": "hr"} for q in st.session_state.hr_questions]

    # show current question
    idx = st.session_state.current_q_index
    total = len(all_qs) + len(st.session_state.voice_questions)
    st.write(f"Question {idx+1} / {total}")

    # If still in text questions
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
                # evaluate using LLM if available, else fallback
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
                st.rerun()
    else:
        # Voice questions stage: index into voice_questions via offset
        v_idx = idx - len(all_qs)
        if v_idx < len(st.session_state.voice_questions):
            st.subheader(f"Q (voice) — {v_idx+1} of {len(st.session_state.voice_questions)}")
            st.write(st.session_state.voice_questions[v_idx])
            st.markdown(
                "Record your answer on your phone/PC and upload the audio file (wav/m4a/mp3). "
                "If you have OpenAI key, the app will attempt Whisper transcription automatically."
            )
            uploaded_audio = st.file_uploader(f"Upload audio answer for voice Q{v_idx+1}", type=["wav","m4a","mp3","ogg"], key=f"audio_{v_idx}")
            if uploaded_audio:
                # save temp file
                tmp_path = DATA_DIR / f"voice_q{v_idx+1}_{st.session_state.candidate.get('name','candidate')}.wav"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                st.success("Audio saved. Attempting transcription...")
                transcript = None
                if os.getenv("OPENAI_API_KEY"):
                    import openai
                    try:
                        openai.api_key = os.getenv("OPENAI_API_KEY")
                        # Whisper transcription (classic)
                        with open(tmp_path, "rb") as af:
                            resp = openai.Audio.transcribe("whisper-1", af)
                        transcript = resp.get("text", "").strip()
                    except Exception as e:
                        st.warning(f"Whisper transcription failed: {e}. Please paste transcript manually.")
                else:
                    st.info("OpenAI key not found — please paste transcript manually.")

                if transcript:
                    st.text_area("Transcript (edit if needed)", value=transcript, key=f"trans_{v_idx}", height=150)
                    if st.button("Submit voice answer (use transcript)", key=f"submit_trans_{v_idx}"):
                        # evaluate
                        try:
                            score, remarks = evaluate_answer_with_llm(st.session_state.voice_questions[v_idx], transcript, "voice", st.session_state.role)
                        except Exception as e:
                            st.warning(f"Evaluation error: {e}. Using heuristic fallback.")
                            length_score = min(5, max(1, len(transcript.split()) // 30))
                            skill_hits = sum(1 for s in st.session_state.skills if s.lower() in transcript.lower())
                            score = min(5, length_score + skill_hits)
                            remarks = "Fallback evaluation applied."
                        st.session_state.voice_answers[v_idx] = tmp_path
                        st.session_state.voice_transcripts[v_idx] = transcript
                        st.session_state.qa_history.append({
                            "question": st.session_state.voice_questions[v_idx],
                            "type": "voice",
                            "answer": transcript,
                            "score": float(score),
                            "remarks": remarks
                        })
                        st.session_state.current_q_index += 1
                        st.rerun()
                else:
                    # no automatic transcript: ask user to paste transcript
                    manual = st.text_area("Paste transcript of your audio here", key=f"manual_trans_{v_idx}", height=150)
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
                            st.session_state.voice_answers[v_idx] = tmp_path
                            st.session_state.voice_transcripts[v_idx] = manual
                            st.session_state.qa_history.append({
                                "question": st.session_state.voice_questions[v_idx],
                                "type": "voice",
                                "answer": manual,
                                "score": float(score),
                                "remarks": remarks
                            })
                            st.session_state.current_q_index += 1
                            st.rerun()
        else:
            # finished all questions
            st.success("All questions answered. Generating report...")
            st.session_state.stage = "done"
            st.rerun()

    # Progress sidebar
    st.sidebar.header("Progress")
    st.sidebar.write(f"Answered: {len(st.session_state.qa_history)}")
    st.sidebar.write(f"Remaining (approx): {total - len(st.session_state.qa_history)}")

# ---------- Done: generate report, rating, eligibility ----------
elif st.session_state.stage == "done":
    st.header("Interview Completed — Report & Eligibility")
    # show QA summary
    for i, qa in enumerate(st.session_state.qa_history, 1):
        st.markdown(f"**{i}. ({qa['type']}) {qa['question']}**")
        st.write(f"Answer: {qa['answer']}")
        st.write(f"Score: {qa['score']:.1f}")
        st.write(f"Remarks: {qa['remarks']}")
        st.markdown("---")

    # compute overall scores
    if st.session_state.qa_history:
        avg_answer_score = sum(q["score"] for q in st.session_state.qa_history) / len(st.session_state.qa_history)
    else:
        avg_answer_score = 0.0
    # resume_score was computed earlier (0-10)
    resume_score = round(min(10.0, max(0.0, (min(6, len(st.session_state.skills)) + min(4, int(st.session_state.get('resume_years',0) or 0)) ))),1)

    # combine into final composite (weighting: resume 40%, answers 60%)
    composite = (resume_score * 0.4) + ((avg_answer_score * 2) * 0.6)  # avg_answer_score is out of 5 -> *2 to scale to 10
    composite = round(composite,1)
    st.metric("Resume score (0-10)", resume_score)
    st.metric("Average answer score (0-5)", round(avg_answer_score,2))
    st.metric("Composite score (0-10)", composite)

    # eligibility decision thresholds
    if composite >= 7.5:
        eligibility = ("Eligible", "green")
    elif composite >= 5.5:
        eligibility = ("Maybe (Requires further interview)", "orange")
    else:
        eligibility = ("Not eligible", "red")

    st.markdown(f"### Eligibility: **{eligibility[0]}**")
    st.info(f"This is an automated recommendation. Final decisions should be made by HR. (Badge color: {eligibility[1]})")

    # generate PDF report (includes chart)
    report_bytes = generate_report_pdf(
        candidate=st.session_state.candidate,
        skills=st.session_state.skills,
        qa_history=st.session_state.qa_history,
        role=st.session_state.role
    )
    st.download_button("Download PDF Report", data=report_bytes, file_name=f"report_{st.session_state.candidate.get('name','candidate')}.pdf", mime="application/pdf")

    if st.button("Restart Interview"):
        keys = ["stage","candidate","resume_text","skills","tech_questions","hr_questions","qa_history","current_q_index","role","voice_questions","voice_answers","voice_transcripts","resume_years","motion_alert_since"]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        # remove flag file
        try:
            if MOTION_FLAG.exists():
                MOTION_FLAG.unlink()
        except Exception:
            pass
        st.rerun()
