import hashlib
import textwrap
from pathlib import Path

import librosa
import numpy as np
import whisper
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Speech Analytics | Track C",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(html):
    """Render HTML safely in Streamlit.

    Strips indentation and blank lines so markdown does not treat the
    HTML as a code block (which is what caused raw tags to appear).
    """
    lines = [line.strip() for line in textwrap.dedent(html).splitlines()]
    st.markdown("\n".join(line for line in lines if line), unsafe_allow_html=True)



# ============================================================
# PROFESSIONAL CSS
# ============================================================

render_html(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(59, 130, 246, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(139, 92, 246, 0.08),
                transparent 28%
            ),
            #07111f;
        color: #e5edf7;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #081321;
        border-right: 1px solid #1d3148;
    }

    section[data-testid="stSidebar"] * {
        color: #dbeafe;
    }

    /* ---------- HEADINGS ---------- */

    h1 {
        font-size: 2.65rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
        color: #f8fafc !important;
    }

    h2 {
        font-size: 1.65rem !important;
        font-weight: 750 !important;
        color: #f8fafc !important;
        margin-top: 1.8rem !important;
    }

    h3 {
        color: #dbeafe !important;
        font-weight: 700 !important;
    }


    /* ---------- HERO ---------- */

    .hero {
        background:
            linear-gradient(
                135deg,
                rgba(30, 64, 175, 0.32),
                rgba(76, 29, 149, 0.25)
            );
        border: 1px solid #294665;
        border-radius: 22px;
        padding: 30px 34px;
        margin-bottom: 28px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.25);
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 850;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #a9bfd8;
        line-height: 1.6;
    }

    .badge {
        display: inline-block;

        padding: 5px 11px;
        border-radius: 999px;
        background: rgba(59,130,246,0.15);
        border: 1px solid rgba(96,165,250,0.35);
        color: #93c5fd;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* ---------- KPI CARDS ---------- */

    .kpi {
        background: linear-gradient(
            145deg,
            #0d1b2b,
            #0a1726
        );
        border: 1px solid #213a55;
        border-radius: 17px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.18);
    }

    .kpi-label {
        color: #8fa9c4;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }


    .kpi-value {
        color: #f8fafc;
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 8px;
    }

    .kpi-blue { color: #60a5fa; }
    .kpi-red { color: #fb7185; }
    .kpi-green { color: #4ade80; }
    .kpi-purple { color: #c084fc; }

    /* ---------- SECTION ---------- */

    .section-label {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .section-number {
        background: #2563eb;
        color: white;
        border-radius: 9px;
        padding: 5px 10px;
        font-weight: 800;
        font-size: 0.82rem;
    }


    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f8fafc;
    }

    /* ---------- INFO CARDS ---------- */

    .info-card {
        background: #0b1929;
        border: 1px solid #203951;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 10px;
    }

    .info-title {
        color: #93c5fd;
        font-size: 0.82rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: .5px;
    }

    .info-value {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 7px;
    }

    /* ---------- EXPLANATION ---------- */


    .explanation {
        background: linear-gradient(
            135deg,
            rgba(30,64,175,.18),
            rgba(37,99,235,.08)
        );
        border: 1px solid #285486;
        border-left: 5px solid #3b82f6;
        border-radius: 15px;
        padding: 22px 24px;
        color: #dbeafe;
        line-height: 1.7;
        font-size: 0.98rem;
    }

    /* ---------- RECOMMENDATION ---------- */

    .recommendation {
        background: linear-gradient(
            135deg,
            rgba(22,101,52,.22),
            rgba(20,83,45,.10)
        );
        border: 1px solid #28633e;
        border-left: 5px solid #22c55e;
        border-radius: 15px;
        padding: 22px 24px;
        color: #dcfce7;
        line-height: 1.7;
        font-size: 0.98rem;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #20344b;
        color: #7188a2;
        text-align: center;
        font-size: 0.82rem;
    }

    /* ---------- FILE UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        background: #0b1929;
        border: 1px solid #243f5b;
        border-radius: 15px;
        padding: 8px;
    }

    /* ---------- AUDIO ---------- */

    audio {
        width: 100%;
    }

    /* ---------- METRICS ---------- */

    [data-testid="stMetric"] {
        background: #0b1929;
        border: 1px solid #213a55;

        padding: 15px;
        border-radius: 14px;
    }

    </style>
    """
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"

IDEAL_DEFAULT = DATASET_DIR / "ideal" / "speech1.wav"
FLAWED_DEFAULT = DATASET_DIR / "flawed" / "flawed1.wav"


# ============================================================
# HELPERS
# ============================================================

def load_audio(path):
    return librosa.load(
        str(path),
        sr=16000,
        mono=True
    )



def feature_analysis(y, sr):

    hop = 512

    rms = librosa.feature.rms(
        y=y,
        frame_length=1024,
        hop_length=hop
    )[0]

    f0, _, _ = librosa.pyin(
        y,
        fmin=70,
        fmax=400,
        sr=sr,
        frame_length=2048,
        hop_length=hop
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        hop_length=hop
    )

    times = librosa.times_like(
        rms,
        sr=sr,
        hop_length=hop
    )


    pitch_clean = f0[~np.isnan(f0)]

    if len(pitch_clean) > 0:
        avg_pitch = float(np.mean(pitch_clean))
        pitch_variation = float(np.std(pitch_clean))
    else:
        avg_pitch = 0.0
        pitch_variation = 0.0

    return {
        "times": times,
        "rms": rms,
        "f0": f0,
        "mfcc": mfcc,
        "avg_pitch": avg_pitch,
        "pitch_variation": pitch_variation,
        "avg_energy": float(np.mean(rms)),
        "duration": float(len(y) / sr),
    }


def pct(a, b):
    return ((b - a) / a * 100.0) if a else 0.0


def highlight_region(
    fig,
    start,
    end,
    color,
    label

):

    if start is None or end is None or end <= start:
        return

    fig.add_vrect(
        x0=start,
        x1=end,
        fillcolor=color,
        opacity=0.16,
        line_width=0,
        annotation_text=label,
        annotation_position="top left",
        annotation_font=dict(
            color=color
        )
    )


def comparison_chart(
    ideal_time,
    ideal_value,
    flawed_time,
    flawed_value,
    title,
    y_title,
    start,
    end
):

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=ideal_time,
            y=ideal_value,
            mode="lines",
            name="Ideal",
            line=dict(
                color="#60a5fa",
                width=2
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=flawed_time,
            y=flawed_value,
            mode="lines",
            name="Participant",
            line=dict(
                color="#fb7185",
                width=2
            )
        )
    )

    highlight_region(
        fig,
        start,
        end,
        "#ef4444",
        "Detected flaw"

    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=18,
                color="#e5edf7"
            )
        ),
        xaxis_title="Time (seconds)",
        yaxis_title=y_title,
        height=430,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#091625",
        margin=dict(
            l=45,
            r=20,
            t=55,
            b=45
        ),
        legend=dict(
            orientation="h",
            y=1.08
        )
    )

    return fig


def build_feedback(rate, energy, pitch_var, mfcc_dist):

    """Create a flaw label, explanation, and recommendation from the
    measured regional changes. The text depends on which features
    actually deviate, so different recordings get different feedback.
    """
    issues, recs = [], []

    if rate >= 20:
        issues.append(f"speech rate was {rate:.1f}% faster than the ideal")
        recs.append("Slow down in this region and insert short pauses between major phrases.")
    elif rate <= -20:
        issues.append(f"speech rate was {abs(rate):.1f}% slower than the ideal")
        recs.append("Keep a steadier pace and avoid long hesitations inside phrases.")

    if energy <= -15:
        issues.append(f"vocal energy dropped {abs(energy):.1f}%")
        recs.append("Project your voice more consistently, especially at the end of phrases.")
    elif energy >= 15:
        issues.append(f"vocal energy rose {energy:.1f}%")
        recs.append("Keep loudness even and avoid sudden volume spikes.")

    if pitch_var <= -20:
        issues.append(f"pitch variation fell {abs(pitch_var):.1f}% (flatter delivery)")
        recs.append("Add pitch movement on key words instead of speaking in a flat tone.")
    elif pitch_var >= 20:
        issues.append(f"pitch variation rose {pitch_var:.1f}%")
        recs.append("Keep intonation controlled and avoid exaggerated pitch swings.")

    if rate >= 20:
        flaw = "Excessive Speech Rate"
    elif rate <= -20:
        flaw = "Insufficient Speech Rate"
    elif pitch_var <= -20:

        flaw = "Flat Prosody"
    elif energy <= -15:
        flaw = "Low Vocal Energy"
    else:
        flaw = "No Clear Flaw"

    if not issues:
        return (
            flaw,
            "No strong acoustic deviation was found in this region.",
            "Maintain the current delivery; no targeted change is needed.",
        )

    explanation = (
        "This region was flagged because " + "; ".join(issues) + ". "
        "These region-specific measurements support the detected delivery deviation."
    )

    return flaw, explanation, " ".join(recs)


# ============================================================
# DYNAMIC TRACK C ANALYSIS
# ============================================================

@st.cache_resource
def get_whisper_model():
    return whisper.load_model("base")


def transcribe_words(path):
    """Return safe Whisper word timestamps for an audio file."""
    model = get_whisper_model()

    try:
        result = model.transcribe(
            str(path),
            word_timestamps=True,
            fp16=False,
        )
    except Exception as e:
        st.warning(f"Whisper transcription failed for {Path(path).name}: {e}")
        return []

    if not isinstance(result, dict):
        return []

    segments = result.get("segments", [])
    if not isinstance(segments, list):
        return []

    words = []

    for seg in segments:
        if not isinstance(seg, dict):
            continue

        segment_words = seg.get("words", [])
        if not isinstance(segment_words, list):
            continue

        for item in segment_words:
            if not isinstance(item, dict):
                continue

            word_text = item.get("word")
            if not isinstance(word_text, str):
                continue

            word_text = word_text.strip().lower()
            if not word_text:
                continue

            start = item.get("start")
            end = item.get("end")

            try:
                start = float(start)
                end = float(end)
            except (TypeError, ValueError):
                continue

            if not np.isfinite(start) or not np.isfinite(end):
                continue

            if end <= start:
                continue

            words.append({
                "word": word_text,
                "start": start,
                "end": end,
            })

    return words


def detect_fast_regions(words):
    """
    Detect sustained local speech acceleration.

    The detector intentionally avoids treating one short Whisper word as a
    flaw. It uses overlapping word windows, a robust local-rate baseline,
    consecutive fast windows, and very small boundary padding.
    """
    if len(words) < 8:
        return []

    starts = np.array([float(w["start"]) for w in words], dtype=float)
    ends = np.array([float(w["end"]) for w in words], dtype=float)
    durations = ends - starts

    valid = durations > 0
    if int(np.sum(valid)) < 8:
        return []

    window = 5
    window_scores = []

    for i in range(0, len(words) - window + 1):
        start = starts[i]
        end = ends[i + window - 1]
        span = end - start

        if span <= 0:
            continue

        rate = window / span * 60.0
        window_scores.append({
            "index": i,
            "start": float(start),
            "end": float(end),
            "rate": float(rate),
        })

    if len(window_scores) < 3:
        return []

    rates = np.array([item["rate"] for item in window_scores], dtype=float)
    baseline = float(np.median(rates))
    mad = float(np.median(np.abs(rates - baseline)))

    if baseline <= 0:
        return []

    # A sustained 20% acceleration is the minimum signal. MAD makes the
    # threshold adapt to naturally variable speaking pace.
    threshold = max(
        baseline * 1.20,
        baseline + max(1.5 * mad, baseline * 0.08),
    )

    fast = rates >= threshold

    # Require at least two consecutive overlapping windows. Because each
    # window contains five words, this represents sustained acceleration.
    runs = []
    run_start = None

    for i, is_fast in enumerate(fast):
        if is_fast and run_start is None:
            run_start = i
        elif not is_fast and run_start is not None:
            if i - run_start >= 2:
                runs.append((run_start, i - 1))
            run_start = None

    if run_start is not None and len(fast) - run_start >= 2:
        runs.append((run_start, len(fast) - 1))

    if not runs:
        return []

    regions = []

    for first_window_index, last_window_index in runs:
        first_window = window_scores[first_window_index]
        last_window = window_scores[last_window_index]

        first_word_index = first_window["index"]
        last_word_index = last_window["index"] + window - 1

        start = float(first_window["start"])
        end = float(last_window["end"])

        # Only include a directly adjacent word when the timing gap is tiny.
        # This replaces the old 1-second expansion.
        if first_word_index > 0:
            previous = words[first_word_index - 1]
            previous_end = float(previous["end"])
            if start - previous_end <= 0.35:
                start = float(previous["start"])

        if last_word_index + 1 < len(words):
            following = words[last_word_index + 1]
            following_start = float(following["start"])
            if following_start - end <= 0.35:
                end = float(following["end"])

        if end - start >= 0.8:
            regions.append({
                "start": round(start, 3),
                "end": round(end, 3),
            })

    # Merge only overlapping or nearly touching detections.
    merged = []
    for region in regions:
        if not merged or region["start"] - merged[-1]["end"] > 0.25:
            merged.append(region.copy())
        else:
            merged[-1]["end"] = max(
                merged[-1]["end"],
                region["end"],
            )

    return merged

def matching_ideal_region(ideal_words, flawed_region_words):
    if not ideal_words or not flawed_region_words:
        return 0.0, float(ideal_words[-1]["end"] if ideal_words else 0.0)
    first = flawed_region_words[0]["word"]
    last = flawed_region_words[-1]["word"]
    start_i = next((i for i, w in enumerate(ideal_words) if w["word"] == first), None)
    if start_i is None:
        return 0.0, float(ideal_words[-1]["end"])
    end_i = None

    for i in range(len(ideal_words) - 1, start_i - 1, -1):
        if ideal_words[i]["word"] == last:
            end_i = i
            break
    if end_i is None:
        end_i = min(start_i + len(flawed_region_words) + 10, len(ideal_words) - 1)
    return float(ideal_words[start_i]["start"]), float(ideal_words[end_i]["end"])


def regional_features(path, start, end, word_count):
    y, sr = load_audio(path)
    start_i = int(max(0, start) * sr)
    end_i = int(min(end, len(y) / sr) * sr)
    segment = y[start_i:end_i]
    duration = max(0.001, end - start)
    rate = word_count / duration * 60.0
    rms = librosa.feature.rms(y=segment)[0] if len(segment) else np.array([0.0])
    energy = float(np.mean(rms))
    try:
        pitch, _, _ = librosa.pyin(segment, fmin=70, fmax=400, sr=sr)
        valid = pitch[~np.isnan(pitch)]
    except Exception:
        valid = np.array([])
    avg_pitch = float(np.mean(valid)) if len(valid) else 0.0
    pitch_var = float(np.std(valid)) if len(valid) else 0.0
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13) if len(segment) else np.zeros((13, 1))
    return {
        "speech_rate": rate,
        "energy": energy,
        "average_pitch": avg_pitch,
        "pitch_variation": pitch_var,
        "mfcc": np.mean(mfcc, axis=1).tolist(),

    }


def dynamic_track_c_analysis(ideal_path, flawed_path):
    ideal_words = transcribe_words(ideal_path)
    flawed_words = transcribe_words(flawed_path)
    regions = detect_fast_regions(flawed_words)

    if regions:
        detected_start = min(r["start"] for r in regions)
        detected_end = max(r["end"] for r in regions)
    else:
        # If no sustained fast-speech region is found, use a small neutral
        # comparison window rather than presenting a large arbitrary portion
        # of the recording as a detected flaw.
        duration = librosa.get_duration(path=str(flawed_path))
        window = min(3.0, max(1.0, duration * 0.20))
        detected_start = max(0.0, (duration - window) / 2.0)
        detected_end = min(duration, detected_start + window)

    flawed_region_words = [
        w for w in flawed_words
        if w["end"] >= detected_start and w["start"] <= detected_end
    ]
    ideal_start, ideal_end = matching_ideal_region(ideal_words, flawed_region_words)
    ideal_region_words = [
        w for w in ideal_words
        if w["end"] >= ideal_start and w["start"] <= ideal_end
    ]

    ideal_f = regional_features(ideal_path, ideal_start, ideal_end, len(ideal_region_words))
    flawed_f = regional_features(flawed_path, detected_start, detected_end, len(flawed_region_words))

    rate_change = pct(ideal_f["speech_rate"], flawed_f["speech_rate"])
    energy_change = pct(ideal_f["energy"], flawed_f["energy"])

    pitch_change = pct(ideal_f["average_pitch"], flawed_f["average_pitch"])
    pitch_var_change = pct(ideal_f["pitch_variation"], flawed_f["pitch_variation"])
    mfcc_dist = float(np.linalg.norm(np.array(ideal_f["mfcc"]) - np.array(flawed_f["mfcc"])))

    score = 0
    if rate_change >= 50:
        score += 50
    elif rate_change >= 30:
        score += 40
    elif rate_change >= 20:
        score += 25
    if rate_change >= 50:
        score += 20
    elif rate_change >= 30:
        score += 15
    elif rate_change >= 20:
        score += 10
    if energy_change <= -30:
        score += 15
    elif energy_change <= -15:
        score += 10
    elif energy_change <= -5:
        score += 5
    if pitch_var_change <= -30:
        score += 10
    elif pitch_var_change <= -20:
        score += 7
    elif pitch_var_change <= -10:
        score += 4
    score = min(score, 100)

    severity = "HIGH" if score >= 70 else ("MEDIUM" if score >= 40 else "LOW")

    evidence_count = 0
    if abs(rate_change) >= 20:
        evidence_count += 1
    if abs(energy_change) >= 15:
        evidence_count += 1
    if abs(pitch_var_change) >= 20:
        evidence_count += 1
    if mfcc_dist >= 25:
        evidence_count += 1

    confidence = min(
        0.98,
        0.50
        + min(abs(rate_change) / 250.0, 0.25)
        + min(abs(pitch_var_change) / 250.0, 0.15)
        + evidence_count * 0.03,
    )

    flaw, explanation, recommendation = build_feedback(
        rate_change, energy_change, pitch_var_change, mfcc_dist
    )

    return {
        "flaw_type": flaw,
        "severity": severity,
        "causal_score": score,
        "confidence": confidence,
        "detected_start": detected_start,
        "detected_end": detected_end,
        # Ground truth is not known for uploaded audio. It is provided by the
        # user in Evaluation mode. Do not copy the detection here, or the
        # evaluation would always score 100%.
        "ground_truth_start": None,
        "ground_truth_end": None,
        "regional_speech_rate_change": rate_change,
        "regional_energy_change": energy_change,
        "regional_pitch_change": pitch_change,
        "regional_pitch_variation_change": pitch_var_change,
        "regional_mfcc_distance": mfcc_dist,
        "causal_explanation": explanation,
        "recommendation": recommendation,
        "dynamic": True,
    }


@st.cache_data(show_spinner=False)
def cached_analysis(ideal_path, flawed_path, stamp):
    # `stamp` changes when the uploaded audio changes, so the cache refreshes.

    return dynamic_track_c_analysis(Path(ideal_path), Path(flawed_path))


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div class="hero">

        <div class="badge">
            TRACK C • MULTIMODAL AI HACKATHON
        </div>

        <div class="hero-title">
            🎙️ Contrastive Speech Analytics
        </div>

        <div class="hero-subtitle">
            Temporal flaw grounding • Acoustic analysis •
            Causal explainability • Actionable feedback
        </div>

    </div>
    """
)


# ============================================================
# SIDEBAR
# ============================================================


with st.sidebar:

    render_html(
        """
        <div style="
            font-size:1.35rem;
            font-weight:800;
            color:#f8fafc;
            margin-bottom:8px;
        ">
        🎙️ SpeechLab
        </div>

        <div style="
            color:#7891ad;
            font-size:.82rem;
            margin-bottom:25px;
        ">
        Track C Analysis Platform
        </div>
        """
    )

    mode = st.radio(
        "Analysis Mode",
        [
            "Prediction / Analysis",
            "Evaluation / Ground Truth"
        ]
    )


    if mode == "Evaluation / Ground Truth":
        st.markdown("### Ground-truth interval")
        gt_start_input = st.number_input(
            "Ground-truth start (s)", min_value=0.0, value=0.0, step=0.01
        )
        gt_end_input = st.number_input(
            "Ground-truth end (s)", min_value=0.0, value=0.0, step=0.01
        )

    st.divider()

    st.markdown("### Dataset")

    st.code(
        "dataset/\n"
        "├── ideal/\n"
        "│   └── speech1.wav\n"
        "└── flawed/\n"
        "    └── flawed1.wav",
        language="text"
    )

    st.caption(
        "Upload both WAV files to run live analysis. "
        "Without uploads, the default dataset files are used."
    )


# ============================================================
# INPUT
# ============================================================


render_html(
    """
    <div class="section-label">
        <span class="section-number">01</span>
        <span class="section-title">Speech Input</span>
    </div>
    """
)

c1, c2 = st.columns(2)

with c1:

    st.markdown("### 🎯 Ideal / Baseline")

    ideal_upload = st.file_uploader(
        "Upload ideal WAV",
        type=["wav"],
        key="ideal_upload",
        label_visibility="collapsed"
    )

with c2:

    st.markdown("### 🧑 Participant / Flawed")

    flawed_upload = st.file_uploader(
        "Upload participant WAV",
        type=["wav"],
        key="flawed_upload",
        label_visibility="collapsed"
    )



# ============================================================
# AUDIO FILE SELECTION
# ============================================================

if ideal_upload:

    ideal_path = BASE_DIR / "_dashboard_ideal.wav"

    with open(ideal_path, "wb") as f:
        f.write(ideal_upload.getvalue())

else:

    ideal_path = IDEAL_DEFAULT


if flawed_upload:

    flawed_path = BASE_DIR / "_dashboard_flawed.wav"

    with open(flawed_path, "wb") as f:
        f.write(flawed_upload.getvalue())

else:

    flawed_path = FLAWED_DEFAULT


if not ideal_path.exists() or not flawed_path.exists():


    st.warning(
        "Please provide both ideal and participant WAV files."
    )

    st.stop()


# ============================================================
# AUDIO PLAYBACK
# ============================================================

render_html(
    """
    <div class="section-label">
        <span class="section-number">02</span>
        <span class="section-title">Audio Playback</span>
    </div>
    """
)

c1, c2 = st.columns(2)

with c1:

    st.markdown("**Ideal Speech**")

    with open(ideal_path, "rb") as f:
        st.audio(
            f.read(),
            format="audio/wav"
        )


with c2:

    st.markdown("**Participant Speech**")

    with open(flawed_path, "rb") as f:
        st.audio(
            f.read(),
            format="audio/wav"
        )


# ============================================================
# AUDIO ANALYSIS
# ============================================================

with st.spinner("Analyzing acoustic features..."):

    ideal_y, ideal_sr = load_audio(ideal_path)
    flawed_y, flawed_sr = load_audio(flawed_path)

    ideal = feature_analysis(ideal_y, ideal_sr)
    flawed = feature_analysis(flawed_y, flawed_sr)


# ============================================================
# RESULT (always computed from the audio pair being shown)
# ============================================================

stamp = (
    (hashlib.md5(ideal_upload.getvalue()).hexdigest() if ideal_upload else "default")
    + "-"
    + (hashlib.md5(flawed_upload.getvalue()).hexdigest() if flawed_upload else "default")

)

with st.spinner("Running Whisper + temporal grounding + acoustic analysis..."):
    result = cached_analysis(str(ideal_path), str(flawed_path), stamp)

flaw_type = result["flaw_type"]
severity = result["severity"]
causal_score = result["causal_score"]
confidence = result["confidence"]

detected_start = result["detected_start"]
detected_end = result["detected_end"]

rate_change = result["regional_speech_rate_change"]
energy_change = result["regional_energy_change"]
pitch_change = result["regional_pitch_change"]
pitch_variation_change = result["regional_pitch_variation_change"]
mfcc_distance = result["regional_mfcc_distance"]

explanation = result["causal_explanation"]
recommendation = result["recommendation"]

# Ground truth only exists when the user enters it in Evaluation mode.
if mode == "Evaluation / Ground Truth":
    ground_start = gt_start_input
    ground_end = gt_end_input
else:
    ground_start = None
    ground_end = None


# ============================================================

# KPI SUMMARY
# ============================================================

render_html(
    """
    <div class="section-label">
        <span class="section-number">03</span>
        <span class="section-title">Analysis Summary</span>
    </div>
    """
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    render_html(
        f"""
        <div class="kpi">
            <div class="kpi-label">Detected Flaw</div>
            <div class="kpi-value kpi-red">{flaw_type}</div>
        </div>
        """
    )

with k2:
    render_html(
        f"""
        <div class="kpi">
            <div class="kpi-label">Severity</div>
            <div class="kpi-value kpi-red">{severity}</div>
        </div>
        """

    )

with k3:
    render_html(
        f"""
        <div class="kpi">
            <div class="kpi-label">Causal Score</div>
            <div class="kpi-value kpi-blue">
                {causal_score:.0f}<span style="font-size:1rem">/100</span>
            </div>
        </div>
        """
    )

with k4:
    render_html(
        f"""
        <div class="kpi">
            <div class="kpi-label">Confidence</div>
            <div class="kpi-value kpi-green">{confidence:.2f}</div>
        </div>
        """
    )


# ============================================================
# ACOUSTIC EVIDENCE
# ============================================================

render_html(
    """
    <div class="section-label">

        <span class="section-number">04</span>
        <span class="section-title">Regional Acoustic Evidence</span>
    </div>
    """
)

m1, m2, m3, m4, m5 = st.columns(5)

metrics = [
    ("Speech Rate", f"{rate_change:+.2f}%", "kpi-red"),
    ("Energy", f"{energy_change:+.2f}%", "kpi-purple"),
    ("Pitch", f"{pitch_change:+.2f}%", "kpi-blue"),
    ("Pitch Variation", f"{pitch_variation_change:+.2f}%", "kpi-red"),
    ("MFCC Distance", f"{mfcc_distance:.2f}", "kpi-purple"),
]

for col, (label, value, color_class) in zip([m1, m2, m3, m4, m5], metrics):

    with col:

        render_html(
            f"""
            <div class="kpi">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {color_class}">{value}</div>
            </div>
            """
        )


# ============================================================
# TEMPORAL GROUNDING

# ============================================================

render_html(
    """
    <div class="section-label">
        <span class="section-number">05</span>
        <span class="section-title">Temporal Flaw Grounding</span>
    </div>
    """
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=ideal["times"],
        y=np.zeros_like(ideal["times"]),
        mode="lines",
        name="Ideal",
        line=dict(color="#60a5fa", width=2)
    )
)

fig.add_trace(
    go.Scatter(
        x=flawed["times"],
        y=np.ones_like(flawed["times"]),
        mode="lines",
        name="Participant",
        line=dict(color="#fb7185", width=2)
    )
)


highlight_region(
    fig,
    detected_start,
    detected_end,
    "#ef4444",
    "Detected flaw"
)

if mode == "Evaluation / Ground Truth":

    highlight_region(
        fig,
        ground_start,
        ground_end,
        "#22c55e",
        "Ground truth"
    )

fig.update_layout(
    title="Where did the system find the problem?",
    xaxis_title="Time (seconds)",
    yaxis=dict(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["Ideal", "Participant"]
    ),
    height=400,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#091625",
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# ACOUSTIC TIME SERIES
# ============================================================

render_html(
    """
    <div class="section-label">
        <span class="section-number">06</span>
        <span class="section-title">Acoustic Time-Series Analysis</span>
    </div>
    """
)

tab1, tab2, tab3 = st.tabs(
    [
        "📈 Pitch / F0",
        "🔊 Energy",
        "🧠 MFCC"
    ]
)


with tab1:

    fig = comparison_chart(

        ideal["times"],
        ideal["f0"],
        flawed["times"],
        flawed["f0"],
        "Pitch / F0 Over Time",
        "Frequency (Hz)",
        detected_start,
        detected_end
    )

    st.plotly_chart(fig, use_container_width=True)


with tab2:

    fig = comparison_chart(
        ideal["times"],
        ideal["rms"],
        flawed["times"],
        flawed["rms"],
        "Vocal Energy Over Time",
        "RMS Energy",
        detected_start,
        detected_end
    )

    st.plotly_chart(fig, use_container_width=True)


with tab3:

    times = librosa.times_like(

        flawed["mfcc"],
        sr=flawed_sr,
        hop_length=512
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=flawed["mfcc"],
            x=times,
            y=[f"MFCC {i+1}" for i in range(flawed["mfcc"].shape[0])],
            colorscale="Viridis"
        )
    )

    fig.update_layout(
        title="Participant MFCC Representation",
        xaxis_title="Time (seconds)",
        height=500,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#091625"
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PITCH VARIATION
# ============================================================

render_html(
    """

    <div class="section-label">
        <span class="section-number">07</span>
        <span class="section-title">Pitch Variation</span>
    </div>
    """
)


def rolling_variation(values, window=20):

    clean = np.nan_to_num(values, nan=0)

    output = np.zeros_like(clean)

    for i in range(window, len(clean)):

        output[i] = np.std(clean[i - window:i])

    return output


ideal_var = rolling_variation(ideal["f0"])

flawed_var = rolling_variation(flawed["f0"])

fig = comparison_chart(
    ideal["times"],
    ideal_var,
    flawed["times"],
    flawed_var,
    "Pitch Variation Over Time",
    "Variation",

    detected_start,
    detected_end
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# FFT
# ============================================================

render_html(
    """
    <div class="section-label">
        <span class="section-number">08</span>
        <span class="section-title">Frequency-Domain Analysis</span>
    </div>
    """
)

spectrum = np.abs(np.fft.rfft(flawed_y))

frequencies = np.fft.rfftfreq(len(flawed_y), d=1 / flawed_sr)

mask = frequencies <= 4000

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=frequencies[mask],
        y=spectrum[mask],

        mode="lines",
        line=dict(color="#a78bfa", width=2),
        name="FFT"
    )
)

fig.update_layout(
    title="Participant Speech FFT Spectrum",
    xaxis_title="Frequency (Hz)",
    yaxis_title="Magnitude",
    height=420,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#091625"
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# CAUSAL EXPLANATION
# ============================================================

render_html(
    """
    <div class="section-label">
        <span class="section-number">09</span>
        <span class="section-title">Causal Explainability</span>
    </div>
    """
)


render_html(
    f"""
    <div class="explanation">
        <strong>Why was this region flagged?</strong><br><br>
        {explanation}
    </div>
    """
)

render_html(
    f"""
    <div class="recommendation">
        <strong>Recommended Action</strong><br><br>
        {recommendation}
    </div>
    """
)


# ============================================================
# TEMPORAL EVALUATION
# ============================================================

render_html(
    """
    <div class="section-label">
        <span class="section-number">10</span>
        <span class="section-title">Temporal Evaluation</span>
    </div>
    """
)


if mode == "Evaluation / Ground Truth":

    if ground_start is None or ground_end is None or ground_end <= ground_start:

        st.warning(
            "Enter a valid ground-truth interval in the sidebar "
            "(end must be greater than start) to compute metrics."
        )

    else:

        intersection_start = max(detected_start, ground_start)
        intersection_end = min(detected_end, ground_end)

        intersection = max(0.0, intersection_end - intersection_start)

        detected_duration = detected_end - detected_start
        ground_duration = ground_end - ground_start

        union = detected_duration + ground_duration - intersection

        iou = intersection / union if union > 0 else 0.0
        precision = intersection / detected_duration if detected_duration > 0 else 0.0
        recall = intersection / ground_duration if ground_duration > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        e1, e2, e3, e4 = st.columns(4)


        e1.metric("Temporal IoU", f"{iou * 100:.2f}%")
        e2.metric("Precision", f"{precision * 100:.2f}%")
        e3.metric("Recall", f"{recall * 100:.2f}%")
        e4.metric("F1 Score", f"{f1 * 100:.2f}%")

        st.caption(
            f"Detected interval: {detected_start:.2f}s – {detected_end:.2f}s. "
            f"Ground truth: {ground_start:.2f}s – {ground_end:.2f}s. "
            "Results are for this single audio pair; expand the dataset before "
            "reporting them as benchmark performance."
        )

else:

    detected_duration = detected_end - detected_start
    share = 100 * detected_duration / flawed["duration"] if flawed["duration"] else 0.0

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Detected Start", f"{detected_start:.2f} s")
    t2.metric("Detected End", f"{detected_end:.2f} s")
    t3.metric("Flagged Duration", f"{detected_duration:.2f} s")
    t4.metric("Share of Audio", f"{share:.1f}%")

    st.caption(
        "Prediction mode: these are the system's temporal predictions. "
        "The detector requires sustained local acceleration before marking "
        "a region. Switch to Evaluation mode and enter a ground-truth "
        "interval to compute IoU, precision, recall, and F1."
    )


# ============================================================
# AUDIO PROFILE

# ============================================================

render_html(
    """
    <div class="section-label">
        <span class="section-number">11</span>
        <span class="section-title">Audio Profile</span>
    </div>
    """
)

c1, c2 = st.columns(2)

with c1:

    render_html(
        f"""
        <div class="info-card">
            <div class="info-title">Ideal / Baseline</div>
            <div class="info-value">{ideal["duration"]:.2f} seconds</div>
            <div style="color:#8fa9c4;margin-top:8px;">
                Sample rate: {ideal_sr} Hz<br>
                Average pitch: {ideal["avg_pitch"]:.2f} Hz<br>
                Average energy: {ideal["avg_energy"]:.5f}<br>
                Pitch variation: {ideal["pitch_variation"]:.2f}
            </div>
        </div>
        """
    )

with c2:


    render_html(
        f"""
        <div class="info-card">
            <div class="info-title">Participant / Flawed</div>
            <div class="info-value">{flawed["duration"]:.2f} seconds</div>
            <div style="color:#8fa9c4;margin-top:8px;">
                Sample rate: {flawed_sr} Hz<br>
                Average pitch: {flawed["avg_pitch"]:.2f} Hz<br>
                Average energy: {flawed["avg_energy"]:.5f}<br>
                Pitch variation: {flawed["pitch_variation"]:.2f}
            </div>
        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.caption("🔄 Results were computed live from the audio pair shown above.")

render_html(
    """
    <div class="footer">
        <strong>Track C • Contrastive Speech Analytics</strong><br>
        Temporal Grounding • Acoustic Evidence • Causal Explainability
        <br><br>
        AI&DS Hackathon Prototype
    </div>
    """
)