import streamlit as st
import pandas as pd
import joblib
import os

# ─────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Football Scout",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: #1c2333; border-radius: 10px; padding: 10px; }
    .block-container { padding-top: 2rem; }
    .style-tag {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 15px;
        margin: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(BASE_DIR, 'xgboost_scouting_model.pkl')
        artifact = joblib.load(path)
        if isinstance(artifact, dict):
            return artifact['model'], artifact['threshold'], artifact['features']
        else:
            default_features = [
                'total_actions', 'avg_x', 'avg_y', 'pressure_events',
                'offensive_score', 'defensive_score',
                'performance_index', 'duels', 'carries'
            ]
            return artifact, 0.65, default_features
    except FileNotFoundError:
        return None, None, None

model, threshold, feature_cols = load_model()

# ─────────────────────────────────────────
# 3. PLAYING STYLE ENGINE
# ─────────────────────────────────────────
def classify_playing_style(offensive_score, defensive_score, carries,
                            duels, pressure_events, total_actions, avg_x):
    """
    Characterizes specific playing styles based on quality and quantity
    of different action types — as stated in the research paper.
    """
    styles = []

    off_ratio = offensive_score / (total_actions + 1) * 100
    def_ratio = defensive_score / (total_actions + 1) * 100
    carry_ratio = carries / (total_actions + 1) * 100
    duel_ratio = duels / (total_actions + 1) * 100
    pressure_ratio = pressure_events / (total_actions + 1) * 100

    # Style 1: Goal Threat
    if off_ratio > 4 and avg_x > 80:
        styles.append(("⚡ Goal Threat",
                        "High offensive output in advanced areas. Primary scoring threat."))

    # Style 2: Creative Playmaker
    if off_ratio > 2.5 and carry_ratio > 12 and avg_x > 55:
        styles.append(("🎨 Creative Playmaker",
                        "Combines ball progression with goal involvement. Creates danger."))

    # Style 3: Defensive Wall
    if def_ratio > 3 and avg_x < 55:
        styles.append(("🛡️ Defensive Wall",
                        "Dominates defensive actions. Protects space and disrupts attacks."))

    # Style 4: Ball Carrier
    if carry_ratio > 15 and duel_ratio > 5:
        styles.append(("🏃 Ball Carrier",
                        "Excels at progressing the ball through carries and winning duels."))

    # Style 5: Box-to-Box
    if off_ratio > 1.5 and def_ratio > 1.5 and 50 < avg_x < 80:
        styles.append(("🔄 Box-to-Box",
                        "Balanced contribution across the pitch. High work rate and versatility."))

    # Style 6: Pressure Resistant
    if pressure_ratio > 20:
        styles.append(("💪 Pressure Resistant",
                        "Performs effectively under opponent pressure. High mental resilience."))

    # Style 7: Deep Lying
    if avg_x < 45 and def_ratio > 2:
        styles.append(("⚓ Deep Lying Anchor",
                        "Operates deep, providing defensive cover and recycling possession."))

    # Fallback
    if not styles:
        styles.append(("📋 Standard Profile",
                        "Balanced but no dominant style pattern detected in this dataset."))

    return styles


# ─────────────────────────────────────────
# 4. HEADER
# ─────────────────────────────────────────
st.title("⚽ AI Football Scouting Dashboard")
st.caption("XGBoost Classifier — StatsBomb Open Data | Top 5 European Leagues")
st.markdown("---")

# ─────────────────────────────────────────
# 5. SIDEBAR — PLAYER INPUTS
# ─────────────────────────────────────────
st.sidebar.header("🔍 Player Profile")
st.sidebar.caption("Enter total season statistics for the prospect.")

position = st.sidebar.selectbox(
    "Position",
    ["Center Back", "Full Back", "Midfielder", "Winger", "Forward"]
)

player_age = st.sidebar.number_input(
    "Player Age", min_value=15, max_value=40, value=23,
    help="Used to flag promising young talent (≤23 years)."
)

# في الـ sidebar — أضف هذا السطر
matches_played = st.sidebar.number_input(
    "Matches Played", min_value=1, max_value=60, value=38,
    help="For context only — does not affect the model."
)

st.sidebar.markdown("#### 📊 Volume & Positioning")
total_actions   = st.sidebar.number_input("Total Actions (season)", min_value=1,   max_value=6000, value=2000,
                                           help="All passes, duels, shots, carries, etc.")
pressure_events = st.sidebar.number_input("Actions Under Pressure",  min_value=0,   max_value=3000, value=300,
                                           help="Events performed while being pressed by opponent.")
avg_x = st.sidebar.slider("Avg. Pitch Position — Length (X)", 0.0, 120.0, 60.0,
                           help="0 = own goal, 120 = opponent goal.")
avg_y = st.sidebar.slider("Avg. Pitch Position — Width  (Y)", 0.0, 80.0,  40.0,
                           help="0 = right flank, 80 = left flank, 40 = center.")

st.sidebar.markdown("#### ⚔️ Offensive Output")
goals   = st.sidebar.number_input("Goals",   min_value=0, max_value=60, value=5)
assists = st.sidebar.number_input("Assists", min_value=0, max_value=60, value=3)

st.sidebar.markdown("#### 🛡️ Defensive Output")
interceptions = st.sidebar.number_input("Interceptions", min_value=0, max_value=300, value=40,
                                         help="Ball recoveries by reading the game.")
clearances    = st.sidebar.number_input("Clearances",    min_value=0, max_value=300, value=30,
                                         help="Defensive actions to clear danger.")

st.sidebar.markdown("#### 🏃 Ball Progression")
duels   = st.sidebar.number_input("Duels Won",    min_value=0, max_value=400, value=80)
carries = st.sidebar.number_input("Ball Carries", min_value=0, max_value=800, value=250)

# ─────────────────────────────────────────
# 6. PREDICTION ENGINE
# ─────────────────────────────────────────
if st.sidebar.button("🎯 Evaluate Player", type="primary", use_container_width=True):

    if model is None:
        st.error("⚠️ Model file not found. Place 'xgboost_scouting_model.pkl' in the same folder as app.py.")
        st.stop()

    # ── Feature Engineering (matches training pipeline exactly) ──
    offensive_score = (goals * 3) + (assists * 2)
    defensive_score = (interceptions * 1.5) + (clearances * 1)
    performance_index = (
        (offensive_score + defensive_score) / total_actions * 100
        if total_actions > 0 else 0
    )

    input_df = pd.DataFrame({
        'total_actions':     [total_actions],
        'avg_x':             [avg_x],
        'avg_y':             [avg_y],
        'pressure_events':   [pressure_events],
        'offensive_score':   [offensive_score],
        'defensive_score':   [defensive_score],
        'performance_index': [performance_index],
        'duels':             [duels],
        'carries':           [carries],
    })[feature_cols]

    elite_prob = model.predict_proba(input_df)[0][1]
    is_elite   = int(elite_prob >= threshold)

    # ── Playing Style Classification ──
    styles = classify_playing_style(
        offensive_score, defensive_score, carries,
        duels, pressure_events, total_actions, avg_x
    )

    # ── Young Player Flag ──
    is_young = player_age <= 23

    # ─────────────────────────────────────────
    # 7. OUTPUT
    # ─────────────────────────────────────────
    st.subheader("📋 Scouting Evaluation Report")

    # ── Row 1: Verdict + Metrics ──
    col_verdict, col_m1, col_m2, col_m3 = st.columns([2, 1, 1, 1])

    with col_verdict:
        if is_elite:
            st.success("### 🌟 ELITE TARGET")
            if is_young:
                st.success("🌱 **PROMISING YOUNG TALENT** — Age ≤ 23")
        else:
            st.info("### 📊 STANDARD PROSPECT")
            if is_young:
                st.info("🌱 **Young Player** — Monitor development (Age ≤ 23)")

        st.metric("Elite Probability",  f"{elite_prob * 100:.1f}%")
        st.metric("Decision Threshold", f"{threshold:.2f}")

    with col_m1:
        st.metric("Offensive Score",   f"{offensive_score:.1f}",
                  help="Goals×3 + Assists×2")
    with col_m2:
        st.metric("Defensive Score",   f"{defensive_score:.1f}",
                  help="Interceptions×1.5 + Clearances×1")
    with col_m3:
        st.metric("Performance Index", f"{performance_index:.2f}",
                  help="(Off+Def) ÷ Actions × 100")

    st.markdown("---")

    # ── Row 2: Playing Style + Interpretation ──
    col_style, col_interp = st.columns([1, 1])

    with col_style:
        st.markdown("#### 🎭 Playing Style Profile")
        st.caption("Characterized by quality and quantity of action types")
        for tag, description in styles:
            st.markdown(f"**{tag}**")
            st.caption(description)

    with col_interp:
        st.markdown("#### 📝 Scouting Interpretation Report")

        if is_elite:
            age_note = " Exceptional value as a young asset with high development ceiling." if is_young else ""
            st.success(
                f"✅ The performance profile of this **{position}** matches elite-tier attributes "
                f"within the top 20% of positional peers. "
                f"Elite classification confidence: **{elite_prob*100:.1f}%** "
                f"(threshold: {threshold:.2f}).{age_note} "
                f"Transfer negotiation and advanced scouting are recommended."
            )
        else:
            age_note = " Given the player's young age, continued development monitoring is advised." if is_young else ""
            st.warning(
                f"⚠️ The **{position}** profile reflects standard squad-level output. "
                f"Elite probability: **{elite_prob*100:.1f}%** "
                f"(threshold: {threshold:.2f}).{age_note} "
                f"Does not meet high-priority classification criteria at this time."
            )

        # Position-specific guidance
        pos_tips = {
            "Center Back":  "Elite CBs: high clearances + interceptions, avg_x 25–45.",
            "Full Back":    "Elite FBs: high carries + assists, wide avg_y positioning.",
            "Midfielder":   "Elite MFs: balanced off+def scores, avg_x 50–70.",
            "Winger":       "Elite Wingers: high carries + duels, avg_x > 75.",
            "Forward":      "Elite Forwards: high offensive score, avg_x > 85.",
        }
        st.info(f"**Position Benchmark — {position}:** {pos_tips[position]}")

    st.markdown("---")

    # ── Row 3: Profile Chart ──
    st.markdown("#### 📊 Action Profile Breakdown")

    max_vals = {
        "Offensive Score":    120,
        "Defensive Score":    200,
        "Performance Index":  50,
        "Pressure Ratio (%)": 60,
        "Duels Won":          400,
        "Carries":            800,
    }
    raw_vals = {
        "Offensive Score":    offensive_score,
        "Defensive Score":    defensive_score,
        "Performance Index":  performance_index,
        "Pressure Ratio (%)": pressure_events / total_actions * 100,
        "Duels Won":          duels,
        "Carries":            carries,
    }

    chart_df = pd.DataFrame({
        "Metric": list(raw_vals.keys()),
        "Score (normalized %)": [
            min(v / max_vals[k] * 100, 100) for k, v in raw_vals.items()
        ]
    }).set_index("Metric")

    st.bar_chart(chart_df, color="#00b4d8")

else:
    # ── Welcome Screen ──
    st.info("👈 Fill in the player stats in the sidebar, then press **Evaluate Player**.")
    st.markdown("""
    #### How the Model Works
    | Component | Description |
    |---|---|
    | **Justice Formula** | Goals×3 + Assists×2 + Interceptions×1.5 + Clearances×1 |
    | **Performance Index** | (Offensive + Defensive Score) ÷ Total Actions × 100 |
    | **Position Rank** | Player ranked vs peers in same position only |
    | **Elite Threshold** | Top 20% within position = Elite |
    | **Playing Style** | Derived from quality and quantity of action types |
    | **Young Talent Flag** | Age ≤ 23 flagged as promising development prospect |

    #### Model Performance
    | Metric | Score |
    |---|---|
    | Accuracy | 96.02% |
    | ROC-AUC | 0.9941 |
    | Elite Precision | 87% |
    | Elite Recall | 95% |

    > Trained on **422,000+ technical events** from **200 matches** across the Top 5 European Leagues (StatsBomb Open Data).
    """)
