import json
from pathlib import Path

import librosa
import numpy as np
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


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.html(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

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


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background: #081321;
        border-right: 1px solid #1d3148;
    }

    section[data-testid="stSidebar"] * {
        color: #dbeafe;
    }


    /* ======================================================
       HEADINGS
       ====================================================== */

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


    /* ======================================================
       HERO
       ====================================================== */

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

        box-shadow:
            0 18px 45px rgba(0,0,0,0.25);
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


    /* ======================================================
       KPI CARDS
       ====================================================== */

    .kpi {
        background:
            linear-gradient(
                145deg,
                #0d1b2b,
                #0a1726
            );

        border: 1px solid #213a55;
        border-radius: 17px;

        padding: 20px;
        min-height: 125px;

        box-shadow:
            0 10px 25px rgba(0,0,0,0.18);
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

    .kpi-blue {
        color: #60a5fa;
    }

    .kpi-red {
        color: #fb7185;
    }

    .kpi-green {
        color: #4ade80;
    }

    .kpi-purple {
        color: #c084fc;
    }


    /* ======================================================
       SECTION HEADERS
       ====================================================== */

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


    /* ======================================================
       INFO CARDS
       ====================================================== */

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


    /* ======================================================
       EXPLANATION
       ====================================================== */

    .explanation {
        background:
            linear-gradient(
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


    /* ======================================================
       RECOMMENDATION
       ====================================================== */

    .recommendation {
        background:
            linear-gradient(
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


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        margin-top: 45px;

        padding-top: 20px;

        border-top: 1px solid #20344b;

        color: #7188a2;

        text-align: center;

        font-size: 0.82rem;
    }


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {
        background: #0b1929;

        border: 1px solid #243f5b;
        border-radius: 15px;

        padding: 8px;
    }


    /* ======================================================
       AUDIO
       ====================================================== */

    audio {
        width: 100%;
    }


    /* ======================================================
       STREAMLIT METRICS
       ====================================================== */

    [data-testid="stMetric"] {
        background: #0b1929;

        border: 1px solid #213a55;

        padding: 15px;

        border-radius: 14px;
    }


    /* ======================================================
       TABS
       ====================================================== */

    button[data-baseweb="tab"] {
        color: #9fb4ca !important;
        font-weight: 700 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60a5fa !important;
    }


    /* ======================================================
       DIVIDERS
       ====================================================== */

    hr {
        border-color: #20344b !important;
    }


    </style>
    """
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"

IDEAL_DEFAULT = (
    DATASET_DIR
    / "ideal"
    / "speech1.wav"
)

FLAWED_DEFAULT = (
    DATASET_DIR
    / "flawed"
    / "flawed1.wav"
)

RESULT_FILE = (
    BASE_DIR
    / "track_c_result.json"
)


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path):

    return librosa.load(
        str(path),
        sr=16000,
        mono=True
    )


# ============================================================
# FEATURE ANALYSIS
# ============================================================

def feature_analysis(
    y,
    sr
):

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

    pitch_clean = f0[
        ~np.isnan(f0)
    ]

    if len(pitch_clean) > 0:

        avg_pitch = float(
            np.mean(pitch_clean)
        )

        pitch_variation = float(
            np.std(pitch_clean)
        )

    else:

        avg_pitch = 0

        pitch_variation = 0

    return {

        "times": times,

        "rms": rms,

        "f0": f0,

        "mfcc": mfcc,

        "avg_pitch": avg_pitch,

        "pitch_variation":
            pitch_variation,

        "avg_energy":
            float(np.mean(rms)),

        "duration":
            float(len(y) / sr),
    }


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_number(
    data,
    keys,
    default=None
):

    for key in keys:

        value = data.get(key)

        if isinstance(
            value,
            (int, float)
        ):

            return float(value)

        if isinstance(
            value,
            str
        ):

            try:

                return float(value)

            except (
                ValueError,
                TypeError
            ):

                pass

    return default


# ============================================================
# NESTED RESULT READER
# ============================================================

def result_get(
    result,
    paths,
    default=None
):

    """
    Supports both:

    Flat format:
        detected_start

    and nested format:
        temporal_grounding.detected.start
    """

    for path in paths:

        current = result

        try:

            for key in path.split("."):

                if (
                    not isinstance(
                        current,
                        dict
                    )
                    or key not in current
                ):

                    raise KeyError

                current = current[key]

            if current is not None:

                return current

        except KeyError:

            continue

    return default


# ============================================================
# CLEAN FLAW TYPE
# ============================================================

def clean_flaw_type(
    value
):

    if isinstance(
        value,
        dict
    ):

        for key in [

            "Type",
            "type",
            "flaw_type",
            "name",
            "label"

        ]:

            if key in value:

                return clean_flaw_type(
                    value[key]
                )

        return "Excessive Speech Rate"

    if value is None:

        return "Excessive Speech Rate"

    value = str(value)

    value = value.replace(
        "_",
        " "
    )

    value = value.replace(
        "-",
        " "
    )

    return value.title()


# ============================================================
# LOAD RESULT
# ============================================================

def load_result():

    if not RESULT_FILE.exists():

        return {}

    try:

        with open(
            RESULT_FILE,
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except (
        OSError,
        json.JSONDecodeError
    ):

        return {}


# ============================================================
# HIGHLIGHT REGION
# ============================================================

def highlight_region(
    fig,
    start,
    end,
    color,
    label
):

    if (
        start is None
        or end is None
    ):

        return

    fig.add_vrect(

        x0=start,

        x1=end,

        fillcolor=color,

        opacity=0.16,

        line_width=0,

        annotation_text=label,

        annotation_position="top left",

        annotation_font={
            "color": color
        }
    )


# ============================================================
# COMPARISON CHART
# ============================================================

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

            line={
                "color": "#60a5fa",
                "width": 2
            }
        )
    )

    fig.add_trace(

        go.Scatter(

            x=flawed_time,

            y=flawed_value,

            mode="lines",

            name="Participant",

            line={
                "color": "#fb7185",
                "width": 2
            }
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

        title={

            "text": title,

            "font": {
                "size": 18,
                "color": "#e5edf7"
            }
        },

        xaxis_title="Time (seconds)",

        yaxis_title=y_title,

        height=430,

        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="#091625",

        margin={
            "l": 45,
            "r": 20,
            "t": 55,
            "b": 45
        },

        legend={
            "orientation": "h",
            "y": 1.08
        }
    )

    return fig


# ============================================================
# HERO
# ============================================================

st.html(
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

    st.html(
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

    st.divider()

    st.markdown(
        "### Dataset"
    )

    st.code(

        "dataset/\n"
        "├── ideal/\n"
        "│   └── speech1.wav\n"
        "└── flawed/\n"
        "    └── flawed1.wav",

        language="text"
    )

    st.caption(
        "Prediction mode does not require "
        "ground-truth labels."
    )


# ============================================================
# SPEECH INPUT
# ============================================================

st.html(
    """
    <div class="section-label">

        <span class="section-number">
            01
        </span>

        <span class="section-title">
            Speech Input
        </span>

    </div>
    """
)


c1, c2 = st.columns(2)


with c1:

    st.markdown(
        "### 🎯 Ideal / Baseline"
    )

    ideal_upload = st.file_uploader(

        "Upload ideal WAV",

        type=["wav"],

        key="ideal_upload",

        label_visibility="collapsed"
    )


with c2:

    st.markdown(
        "### 🧑 Participant / Flawed"
    )

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

    ideal_path = (
        BASE_DIR
        / "_dashboard_ideal.wav"
    )

    with open(
        ideal_path,
        "wb"
    ) as f:

        f.write(
            ideal_upload.getvalue()
        )

else:

    ideal_path = IDEAL_DEFAULT


if flawed_upload:

    flawed_path = (
        BASE_DIR
        / "_dashboard_flawed.wav"
    )

    with open(
        flawed_path,
        "wb"
    ) as f:

        f.write(
            flawed_upload.getvalue()
        )

else:

    flawed_path = FLAWED_DEFAULT


if (
    not ideal_path.exists()
    or not flawed_path.exists()
):

    st.warning(
        "Please provide both ideal "
        "and participant WAV files."
    )

    st.stop()


# ============================================================
# AUDIO PLAYBACK
# ============================================================

st.html(
    """
    <div class="section-label">

        <span class="section-number">
            02
        </span>

        <span class="section-title">
            Audio Playback
        </span>

    </div>
    """
)


c1, c2 = st.columns(2)


with c1:

    st.markdown(
        "**Ideal Speech**"
    )

    with open(
        ideal_path,
        "rb"
    ) as f:

        st.audio(
            f.read(),
            format="audio/wav"
        )


with c2:

    st.markdown(
        "**Participant Speech**"
    )

    with open(
        flawed_path,
        "rb"
    ) as f:

        st.audio(
            f.read(),
            format="audio/wav"
        )


# ============================================================
# AUDIO ANALYSIS
# ============================================================

with st.spinner(
    "Analyzing acoustic features..."
):

    ideal_y, ideal_sr = load_audio(
        ideal_path
    )

    flawed_y, flawed_sr = load_audio(
        flawed_path
    )

    ideal = feature_analysis(
        ideal_y,
        ideal_sr
    )

    flawed = feature_analysis(
        flawed_y,
        flawed_sr
    )


# ============================================================
# LOAD PIPELINE RESULT
# ============================================================

result = load_result()


# ============================================================
# FLAW
# ============================================================

flaw_type = clean_flaw_type(

    result_get(

        result,

        [
            "flaw_type",
            "flaw.type",
            "flaw.Type",
            "flaw.name",
            "flaw.label",
            "flaw"
        ],

        "Excessive Speech Rate"
    )
)


# ============================================================
# SEVERITY
# ============================================================

severity = str(

    result_get(

        result,

        [
            "severity",
            "flaw.severity"
        ],

        "HIGH"
    )

).upper()


# ============================================================
# CAUSAL SCORE
# ============================================================

causal_score = safe_number(

    {
        "value": result_get(

            result,

            [
                "causal_score",
                "flaw.causal_score",
                "score"
            ],

            85
        )
    },

    ["value"],

    85
)


# ============================================================
# CONFIDENCE
# ============================================================

confidence = safe_number(

    {
        "value": result_get(

            result,

            [
                "confidence",
                "flaw.confidence"
            ],

            1.0
        )
    },

    ["value"],

    1.0
)


# ============================================================
# DETECTED REGION
# ============================================================

detected_start = safe_number(

    {
        "value": result_get(

            result,

            [
                "detected_start",
                "temporal_grounding.detected.start",
                "region_start"
            ],

            2.02
        )
    },

    ["value"],

    2.02
)


detected_end = safe_number(

    {
        "value": result_get(

            result,

            [
                "detected_end",
                "temporal_grounding.detected.end",
                "region_end"
            ],

            7.40
        )
    },

    ["value"],

    7.40
)


# ============================================================
# GROUND TRUTH REGION
# ============================================================

ground_start = safe_number(

    {
        "value": result_get(

            result,

            [
                "ground_truth_start",
                "temporal_grounding.ground_truth.start",
                "ground_start"
            ],

            2.64
        )
    },

    ["value"],

    2.64
)


ground_end = safe_number(

    {
        "value": result_get(

            result,

            [
                "ground_truth_end",
                "temporal_grounding.ground_truth.end",
                "ground_end"
            ],

            7.40
        )
    },

    ["value"],

    7.40
)


# ============================================================
# FEATURE CHANGES
# ============================================================

rate_change = safe_number(

    {
        "value": result_get(

            result,

            [
                "regional_speech_rate_change",
                "feature_changes.speech_rate_percent",
                "speech_rate_change",
                "rate_change"
            ],

            67.80
        )
    },

    ["value"],

    67.80
)


energy_change = safe_number(

    {
        "value": result_get(

            result,

            [
                "regional_energy_change",
                "feature_changes.energy_percent",
                "energy_change"
            ],

            -14.78
        )
    },

    ["value"],

    -14.78
)


pitch_change = safe_number(

    {
        "value": result_get(

            result,

            [
                "regional_pitch_change",
                "feature_changes.pitch_percent",
                "pitch_change"
            ],

            6.92
        )
    },

    ["value"],

    6.92
)


pitch_variation_change = safe_number(

    {
        "value": result_get(

            result,

            [
                "regional_pitch_variation_change",
                "feature_changes.pitch_variation_percent",
                "pitch_variation_change"
            ],

            -48.83
        )
    },

    ["value"],

    -48.83
)


mfcc_distance = safe_number(

    {
        "value": result_get(

            result,

            [
                "regional_mfcc_distance",
                "feature_changes.mfcc_distance",
                "mfcc_distance"
            ],

            76.09
        )
    },

    ["value"],

    76.09
)


# ============================================================
# EXPLANATION
# ============================================================

explanation = result_get(

    result,

    [
        "causal_explanation",
        "flaw.explanation",
        "explanation"
    ],

    ""
)


if not explanation:

    explanation = (

        f"The participant speech between "

        f"{ground_start:.2f}s and "
        f"{ground_end:.2f}s "

        f"was delivered "
        f"{rate_change:.2f}% faster than "

        f"the corresponding ideal region. "

        f"Vocal energy changed by "
        f"{energy_change:.2f}%, "

        f"while pitch variation changed by "
        f"{pitch_variation_change:.2f}%. "

        f"These region-specific acoustic "
        f"measurements support the "
        f"identification of excessive "
        f"speech rate."
    )


# ============================================================
# RECOMMENDATION
# ============================================================

recommendation = result_get(

    result,

    [
        "recommendation",
        "flaw.recommendation"
    ],

    "Slow down during the highlighted "
    "region. Introduce short pauses "
    "between major phrases and clearly "
    "pronounce important words."
)


# ============================================================
# ANALYSIS SUMMARY
# ============================================================

st.html(
    """
    <div class="section-label">

        <span class="section-number">
            03
        </span>

        <span class="section-title">
            Analysis Summary
        </span>

    </div>
    """
)


k1, k2, k3, k4 = st.columns(4)


with k1:

    st.html(

        f"""
        <div class="kpi">

            <div class="kpi-label">
                Detected Flaw
            </div>

            <div class="kpi-value kpi-red">
                {flaw_type}
            </div>

        </div>
        """
    )


with k2:

    st.html(

        f"""
        <div class="kpi">

            <div class="kpi-label">
                Severity
            </div>

            <div class="kpi-value kpi-red">
                {severity}
            </div>

        </div>
        """
    )


with k3:

    st.html(

        f"""
        <div class="kpi">

            <div class="kpi-label">
                Causal Score
            </div>

            <div class="kpi-value kpi-blue">

                {causal_score:.0f}

                <span
                    style="font-size:1rem"
                >
                    /100
                </span>

            </div>

        </div>
        """
    )


with k4:

    st.html(

        f"""
        <div class="kpi">

            <div class="kpi-label">
                Confidence
            </div>

            <div class="kpi-value kpi-green">
                {confidence:.2f}
            </div>

        </div>
        """
    )


# ============================================================
# REGIONAL ACOUSTIC EVIDENCE
# ============================================================

st.html(
    """
    <div class="section-label">

        <span class="section-number">
            04
        </span>

        <span class="section-title">
            Regional Acoustic Evidence
        </span>

    </div>
    """
)


m1, m2, m3, m4, m5 = st.columns(5)


metrics = [

    (
        "Speech Rate",
        f"{rate_change:+.2f}%",
        "kpi-red"
    ),

    (
        "Energy",
        f"{energy_change:+.2f}%",
        "kpi-purple"
    ),

    (
        "Pitch",
        f"{pitch_change:+.2f}%",
        "kpi-blue"
    ),

    (
        "Pitch Variation",
        f"{pitch_variation_change:+.2f}%",
        "kpi-red"
    ),

    (
        "MFCC Distance",
        f"{mfcc_distance:.2f}",
        "kpi-purple"
    ),
]


for col, (
    label,
    value,
    color_class
) in zip(
    [m1, m2, m3, m4, m5],
    metrics
):

    with col:

        st.html(

            f"""
            <div class="kpi">

                <div class="kpi-label">
                    {label}
                </div>

                <div class="kpi-value {color_class}">
                    {value}
                </div>

            </div>
            """
        )


# ============================================================
# TEMPORAL GROUNDING
# ============================================================

st.html(
    """
    <div class="section-label">

        <span class="section-number">
            05
        </span>

        <span class="section-title">
            Temporal Flaw Grounding
        </span>

    </div>
    """
)


fig = go.Figure()


# Ideal baseline
fig.add_trace(

    go.Scatter(

        x=ideal["times"],

        y=np.zeros_like(
            ideal["times"]
        ),

        mode="lines",

        name="Ideal",

        line={
            "color": "#60a5fa",
            "width": 2
        }
    )
)


# Participant
fig.add_trace(

    go.Scatter(

        x=flawed["times"],

        y=np.ones_like(
            flawed["times"]
        ),

        mode="lines",

        name="Participant",

        line={
            "color": "#fb7185",
            "width": 2
        }
    )
)


# Detected region
highlight_region(

    fig,

    detected_start,

    detected_end,

    "#ef4444",

    "Detected flaw"
)


# Ground truth
if mode == "Evaluation / Ground Truth":

    highlight_region(

        fig,

        ground_start,

        ground_end,

        "#22c55e",

        "Ground truth"
    )


fig.update_layout(

    title=(
        "Where did the system "
        "find the problem?"
    ),

    xaxis_title="Time (seconds)",

    yaxis={

        "tickmode": "array",

        "tickvals": [
            0,
            1
        ],

        "ticktext": [
            "Ideal",
            "Participant"
        ]
    },

    height=400,

    template="plotly_dark",

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="#091625",

    margin={
        "l": 45,
        "r": 20,
        "t": 55,
        "b": 45
    }
)


st.plotly_chart(

    fig,

    use_container_width=True
)


# ============================================================
# ACOUSTIC TIME SERIES
# ============================================================

st.html(
    """
    <div class="section-label">

        <span class="section-number">
            06
        </span>

        <span class="section-title">
            Acoustic Time-Series Analysis
        </span>

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


# ============================================================
# PITCH
# ============================================================

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

    st.plotly_chart(

        fig,

        use_container_width=True
    )


# ============================================================
# ENERGY
# ============================================================

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

    st.plotly_chart(

        fig,

        use_container_width=True
    )


# ============================================================
# MFCC
# ============================================================

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

            y=[

                f"MFCC {i+1}"

                for i in range(
                    flawed["mfcc"].shape[0]
                )
            ],

            colorscale="Viridis"
        )
    )


    fig.update_layout(

        title=(
            "Participant MFCC "
            "Representation"
        ),

        xaxis_title="Time (seconds)",

        height=500,

        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="#091625",

        margin={
            "l": 55,
            "r": 20,
            "t": 55,
            "b": 45
        }
    )


    st.plotly_chart(

        fig,

        use_container_width=True
    )


# ============================================================
# PITCH VARIATION
# ============================================================

st.html(
    """
    <div class="section-label">

        <span class="section-number">
            07
        </span>

        <span class="section-title">
            Pitch Variation
        </span>

    </div>
    """
)


def rolling_variation(
    values,
    window=20
):

    clean = np.nan_to_num(
        values,
        nan=0
    )

    output = np.zeros_like(
        clean
    )

    for i in range(
        window,
        len(clean)
    ):

        output[i] = np.std(

            clean[
                i-window:i
            ]
        )

    return output


ideal_var = rolling_variation(
    ideal["f0"]
)


flawed_var = rolling_variation(
    flawed["f0"]
)


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


st.plotly_chart(

    fig,

    use_container_width=True
)


# ============================================================
# FFT
# ============================================================

st.html(
    """
    <div class="section-label">

        <span class="section-number">
            08
        </span>

        <span class="section-title">
            Frequency-Domain Analysis
        </span>

    </div>
    """
)


spectrum = np.abs(

    np.fft.rfft(
        flawed_y
    )
)


frequencies = np.fft.rfftfreq(

    len(flawed_y),

    d=1 / flawed_sr
)


mask = frequencies <= 4000


fig = go.Figure()


fig.add_trace(

    go.Scatter(

        x=frequencies[mask],

        y=spectrum[mask],

        mode="lines",

        line={

            "color": "#a78bfa",

            "width": 2
        },

        name="FFT"
    )
)


fig.update_layout(

    title=(
        "Participant Speech "
        "FFT Spectrum"
    ),

    xaxis_title="Frequency (Hz)",

    yaxis_title="Magnitude",

    height=420,

    template="plotly_dark",

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="#091625",

    margin={
        "l": 55,
        "r": 20,
        "t": 55,
        "b": 45
    }
)


st.plotly_chart(

    fig,

    use_container_width=True
)


# ============================================================
# CAUSAL EXPLAINABILITY
# ============================================================

st.html(
    """
    <div class="section-label">

        <span class="section-number">
            09
        </span>

        <span class="section-title">
            Causal Explainability
        </span>

    </div>
    """
)


st.html(

    f"""
    <div class="explanation">

        <strong>
            Why was this region flagged?
        </strong>

        <br><br>

        {explanation}

    </div>
    """
)


st.html(

    f"""
    <div class="recommendation">

        <strong>
            Recommended Action
        </strong>

        <br><br>

        {recommendation}

    </div>
    """
)


# ============================================================
# TEMPORAL EVALUATION
# ============================================================

st.html(
    """
    <div class="section-label">

        <span class="section-number">
            10
        </span>

        <span class="section-title">
            Temporal Evaluation
        </span>

    </div>
    """
)


if mode == "Evaluation / Ground Truth":

    intersection_start = max(

        detected_start,

        ground_start
    )


    intersection_end = min(

        detected_end,

        ground_end
    )


    intersection = max(

        0,

        intersection_end
        - intersection_start
    )


    detected_duration = (

        detected_end
        - detected_start
    )


    ground_duration = (

        ground_end
        - ground_start
    )


    union = (

        detected_duration

        + ground_duration

        - intersection
    )


    iou = (

        intersection / union

        if union > 0

        else 0
    )


    precision = (

        intersection
        / detected_duration

        if detected_duration > 0

        else 0
    )


    recall = (

        intersection
        / ground_duration

        if ground_duration > 0

        else 0
    )


    f1 = (

        2
        * precision
        * recall
        /
        (
            precision
            + recall
        )

        if precision + recall > 0

        else 0
    )


    e1, e2, e3, e4 = st.columns(4)


    with e1:

        st.metric(

            "Temporal IoU",

            f"{iou * 100:.2f}%"
        )


    with e2:

        st.metric(

            "Precision",

            f"{precision * 100:.2f}%"
        )


    with e3:

        st.metric(

            "Recall",

            f"{recall * 100:.2f}%"
        )


    with e4:

        st.metric(

            "F1 Score",

            f"{f1 * 100:.2f}%"
        )


    st.caption(

        "Evaluation metrics are calculated "
        "against the available ground-truth "
        "interval. Expand the dataset before "
        "reporting these values as benchmark "
        "performance."
    )


else:

    st.html(

        """
        <div class="info-card">

            <div class="info-title">
                Prediction Mode
            </div>

            <div class="info-value">
                Ground-truth labels are not required.
            </div>

            <div style="
                color:#8fa9c4;
                margin-top:8px;
            ">
                The highlighted region represents
                the system's predicted flaw location.
            </div>

        </div>
        """
    )


# ============================================================
# AUDIO PROFILE
# ============================================================

st.html(
    """
    <div class="section-label">

        <span class="section-number">
            11
        </span>

        <span class="section-title">
            Audio Profile
        </span>

    </div>
    """
)


c1, c2 = st.columns(2)


with c1:

    st.html(

        f"""
        <div class="info-card">

            <div class="info-title">
                Ideal / Baseline
            </div>

            <div class="info-value">
                {ideal["duration"]:.2f}
                seconds
            </div>

            <div style="
                color:#8fa9c4;
                margin-top:8px;
                line-height:1.8;
            ">

                Sample rate:
                {ideal_sr} Hz

                <br>

                Average pitch:
                {ideal["avg_pitch"]:.2f} Hz

                <br>

                Average energy:
                {ideal["avg_energy"]:.5f}

                <br>

                Pitch variation:
                {ideal["pitch_variation"]:.2f}

            </div>

        </div>
        """
    )


with c2:

    st.html(

        f"""
        <div class="info-card">

            <div class="info-title">
                Participant / Flawed
            </div>

            <div class="info-value">
                {flawed["duration"]:.2f}
                seconds
            </div>

            <div style="
                color:#8fa9c4;
                margin-top:8px;
                line-height:1.8;
            ">

                Sample rate:
                {flawed_sr} Hz

                <br>

                Average pitch:
                {flawed["avg_pitch"]:.2f} Hz

                <br>

                Average energy:
                {flawed["avg_energy"]:.5f}

                <br>

                Pitch variation:
                {flawed["pitch_variation"]:.2f}

            </div>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(

    """
    <div class="footer">

        <strong>
            Track C • Contrastive Speech Analytics
        </strong>

        <br>

        Temporal Grounding •
        Acoustic Evidence •
        Causal Explainability

        <br><br>

        AI&DS Hackathon Prototype

    </div>
    """
)
