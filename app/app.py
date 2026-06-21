"""
Streamlit web application for the Euroleague game outcome predictor.
Run with: streamlit run app/app.py
"""
import sys
import os
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.predict import predict_game, get_available_values, get_team_stats, load_best_model

APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="EuroLeague Predictor",
    page_icon="🟠",
    layout="wide",
)


def _set_bg(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext  = "png" if path.endswith(".png") else "jpeg"
    st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/{ext};base64,{data}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
</style>""", unsafe_allow_html=True)


# ── Background ────────────────────────────────────────────────────────────────
for _bg in ("slika.png", "slika.jpg"):
    _p = os.path.join(APP_DIR, _bg)
    if os.path.exists(_p):
        _set_bg(_p)
        break

# ── Dark panel style for columns ──────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="column"] {
    background: rgba(8, 8, 20, 0.82);
    border-radius: 16px;
    padding: 28px 24px !important;
}
/* keep header outside panels */
.stImage img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Header (full width, logo + text inline) ───────────────────────────────────
_logo_path = os.path.join(APP_DIR, "logo.png")
_logo_html = ""
if os.path.exists(_logo_path):
    with open(_logo_path, "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode()
    _logo_html = f'<img src="data:image/png;base64,{_logo_b64}" width="90" style="border-radius:8px; display:block;">'

st.markdown(f"""
<div style="display:flex; align-items:center; gap:10px; padding:10px 0 18px 0;">
  {_logo_html}
  <div>
    <div style="font-size:30px;font-weight:800;color:white;line-height:1;letter-spacing:-0.5px;">
      EUROLEAGUE <span style="color:#F47920;">PREDICTOR</span>
    </div>
    <div style="font-size:11px;color:#aaa;letter-spacing:3px;text-transform:uppercase;margin-top:5px;">
      Game Outcome Prediction
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _load():
    av = get_available_values()
    _, _, _, _, model_name = load_best_model()
    return av, model_name


try:
    available, model_name = _load()
    teams    = available['teams']
    phases   = available['phases'] or ['REGULAR SEASON', 'PLAYOFFS', 'FINAL FOUR']
    years    = available.get('years', list(range(2007, 2026)))
    clusters = available['team_clusters']
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.info("Run the full pipeline first:\n```\npython main.py all\n```")
    st.stop()

# ── Two-column layout ─────────────────────────────────────────────────────────
left_col, right_col = st.columns(2, gap="large")

# ═══════════════════════════════ LEFT — inputs ════════════════════════════════
with left_col:
    st.markdown(f"<div style='color:#aaa;font-size:12px;margin-bottom:12px;'>Model: <b style='color:white'>{model_name.replace('_',' ').title()}</b></div>", unsafe_allow_html=True)

    st.subheader("Home Team")
    home_team = st.selectbox("Home team", teams, label_visibility="collapsed", key="ht")

    st.subheader("Away Team")
    away_options = [t for t in teams if t != home_team]
    away_team = st.selectbox("Away team", away_options, label_visibility="collapsed", key="at")

    c3, c4 = st.columns(2)
    with c3:
        phase = st.selectbox("Phase", phases)
    with c4:
        year = st.selectbox("Season", years, index=len(years) - 1 if years else 0)

    # Stats cards
    stats = get_team_stats(home_team, away_team, year=int(year), clusters=clusters)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(f"""
<div style="background:rgba(33,150,243,0.18);border-left:3px solid #2196F3;
            padding:10px 12px;border-radius:8px;text-align:center;">
  <div style="color:#aaa;font-size:9px;text-transform:uppercase;letter-spacing:1px;">Forma dom.</div>
  <div style="color:white;font-size:22px;font-weight:700;">{stats['home_form']:.1f}</div>
  <div style="color:#888;font-size:10px;">pts avg</div>
</div>""", unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
<div style="background:rgba(255,87,34,0.18);border-left:3px solid #FF5722;
            padding:10px 12px;border-radius:8px;text-align:center;">
  <div style="color:#aaa;font-size:9px;text-transform:uppercase;letter-spacing:1px;">Forma gost.</div>
  <div style="color:white;font-size:22px;font-weight:700;">{stats['away_form']:.1f}</div>
  <div style="color:#888;font-size:10px;">pts avg</div>
</div>""", unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""
<div style="background:rgba(244,121,32,0.18);border-left:3px solid #F47920;
            padding:10px 12px;border-radius:8px;text-align:center;">
  <div style="color:#aaa;font-size:9px;text-transform:uppercase;letter-spacing:1px;">H2H dom.</div>
  <div style="color:white;font-size:22px;font-weight:700;">{stats['h2h']}</div>
  <div style="color:#888;font-size:10px;">pobjeda</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if st.button("Predict", type="primary", width="stretch"):
        with st.spinner("Calculating..."):
            res = predict_game(
                hometeam=home_team, awayteam=away_team,
                phase=str(phase), year=int(year),
                home_form=stats['home_form'],
                away_form=stats['away_form'],
                h2h=stats['h2h'],
                clusters=clusters,
            )
        st.session_state['result']    = res
        st.session_state['home_team'] = home_team
        st.session_state['away_team'] = away_team

# ═══════════════════════════════ RIGHT — results ══════════════════════════════
with right_col:
    res = st.session_state.get('result')

    if res is None:
        st.markdown("""
<div style="text-align:center;padding:60px 0;color:#555;">
  <div style="font-size:52px;margin-bottom:16px;">🏆</div>
  <div style="font-size:16px;color:#888;">Izaberi timove i klikni<br><b style='color:#F47920'>Predict</b></div>
</div>""", unsafe_allow_html=True)
    else:
        ht   = st.session_state['home_team']
        at   = st.session_state['away_team']
        hp   = res['home_win_probability']
        ap   = res['away_win_probability']
        pred = res['prediction']

        winner      = ht if pred == 'Home Win' else at
        winner_prob = hp if pred == 'Home Win' else ap
        loser_prob  = ap if pred == 'Home Win' else hp

        # Winner banner
        st.markdown(f"""
<div style="text-align:center;padding:20px 0 10px 0;">
  <div style="color:#F47920;font-size:11px;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;">
    Predviđeni pobjednik
  </div>
  <div style="color:white;font-size:26px;font-weight:800;line-height:1.2;">
    {winner}
  </div>
  <div style="color:#F47920;font-size:42px;font-weight:900;margin:4px 0;">
    {winner_prob:.1%}
  </div>
  <div style="color:#888;font-size:12px;">vjerovatnoća pobjede</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Home vs Away breakdown
        st.markdown(f"""
<div style="background:rgba(255,255,255,0.05);border-radius:12px;padding:18px 20px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
    <div style="text-align:left;">
      <div style="color:#aaa;font-size:10px;text-transform:uppercase;letter-spacing:1px;">🏠 Domaćin</div>
      <div style="color:white;font-size:15px;font-weight:600;margin-top:2px;">{ht}</div>
      <div style="color:#2196F3;font-size:28px;font-weight:800;">{hp:.1%}</div>
    </div>
    <div style="color:#555;font-size:22px;font-weight:300;">vs</div>
    <div style="text-align:right;">
      <div style="color:#aaa;font-size:10px;text-transform:uppercase;letter-spacing:1px;">Gost ✈️</div>
      <div style="color:white;font-size:15px;font-weight:600;margin-top:2px;">{at}</div>
      <div style="color:#FF5722;font-size:28px;font-weight:800;">{ap:.1%}</div>
    </div>
  </div>
  <div style="height:8px;background:#222;border-radius:4px;overflow:hidden;">
    <div style="width:{hp*100:.1f}%;height:100%;background:linear-gradient(90deg,#2196F3,#1565C0);
                border-radius:4px;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:4px;">
    <span style="color:#2196F3;font-size:10px;">{hp:.1%}</span>
    <span style="color:#FF5722;font-size:10px;">{ap:.1%}</span>
  </div>
</div>""", unsafe_allow_html=True)
