# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageStat
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from fpdf import FPDF
import base64, io, os, datetime, smtplib
from email.message import EmailMessage
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
# Paths (adjust if necessary)
SOIL_CSV = "/mnt/data/Soil_Nutrients.csv"
CROP_CSV = "/mnt/data/Crop_recommendation.csv"
FERT_CSV = "/mnt/data/fertilizer.csv"
MODELS_DIR = os.path.join(os.getcwd(), "models")
SOIL_MODEL_PATH = os.path.join(MODELS_DIR, "soil_model.h5")
CROP_MODEL_PATH = os.path.join(MODELS_DIR, "crop_model.h5")
FERT_MODEL_PATH = os.path.join(MODELS_DIR, "fertilizer_model.h5")
ASSETS_DIR = os.path.join(os.getcwd(), "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.jpg")

# Email config (optional)
OWNER_EMAIL = ""    # set to your email if using help email
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = ""
SMTP_PASS = ""

st.set_page_config(page_title="AgriVision AI", page_icon="🌾", layout="centered")

# ---------- HELPERS: Read datasets robustly ----------
@st.cache_data
def load_csv_safe(path):
    if not os.path.exists(path):
        return None
    # try reading and show columns
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding="latin1")
    return df

soil_df = load_csv_safe(SOIL_CSV)
crop_df = load_csv_safe(CROP_CSV)
fert_df = load_csv_safe(FERT_CSV)

# ---------- ML model loader (optional) ----------
def load_model_if_exists(path):
    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path)
            return model
        except Exception as e:
            st.warning(f"Could not load model at {path}: {e}")
    return None

soil_model = load_model_if_exists(SOIL_MODEL_PATH)
crop_model = load_model_if_exists(CROP_MODEL_PATH)
fert_model = load_model_if_exists(FERT_MODEL_PATH)

# ---------- UI style tweaks ----------
# Increase sidebar font and adjust radio options with custom CSS
st.markdown("""
    <style>
    /* Sidebar title / radio larger */
    .css-1d391kg {font-size:18px !important;}  /* streamlit class for radio labels (may vary) */
    .sidebar .stRadio {font-size:18px !important;}
    .sidebar .css-1emrehy {font-size:18px !important;}
    /* Make header look professional */
    .app-header {display:flex; align-items:center;}
    </style>
    """, unsafe_allow_html=True)

# Top header
c1, c2 = st.columns([1,6])
with c1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=140)
with c2:
    st.markdown("<h1 style='margin-bottom:0'>AgriVision AI</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:gray'>Smart Farming Decision & Analysis System</div>", unsafe_allow_html=True)

# Sidebar navigation (bigger labels via markdown)
st.sidebar.markdown("<h3 style='font-size:20px'>Navigation</h3>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Home","Detection & Suggestion","ChatBot","Help Centre","Downloads"], index=0)

# ---------- UTIL: soil estimation from soil_df or heuristics ----------
def estimate_npk_ph_from_inputs(soil_type=None, region=None, area_type="Humid", rainfall="Medium"):
    # If we have soil_df and it contains N,P,K,pH use it
    if isinstance(soil_df, pd.DataFrame):
        # try to find matching region or soil type
        if region and region in soil_df.columns:
            pass
        # common columns: try to detect column names
        cols = [c.lower() for c in soil_df.columns]
        # Match candidate columns
        def find(col_names):
            for name in col_names:
                for c in soil_df.columns:
                    if name in c.lower():
                        return c
            return None
        colN = find(["n","nitrogen"])
        colP = find(["p","phosphorus"])
        colK = find(["k","potassium"])
        colph = find(["ph"])
        # If crop recommendation CSV has NPK values per sample, we can use median by soil type if soil_type exists
        if colN and colP and colK and colph:
            # attempt selection by soil_type or use median
            if soil_type and 'soil_type' in [c.lower() for c in soil_df.columns]:
                try:
                    sel = soil_df[soil_df['soil_type'].str.lower() == soil_type.lower()]
                    if not sel.empty:
                        N = float(sel[colN].median()); P = float(sel[colP].median()); K = float(sel[colK].median()); pH = float(sel[colph].median())
                        return round(N,1), round(P,1), round(K,1), round(pH,2)
                except Exception:
                    pass
            # fallback to medians
            try:
                N = float(soil_df[colN].median()); P = float(soil_df[colP].median()); K = float(soil_df[colK].median()); pH = float(soil_df[colph].median())
                return round(N,1), round(P,1), round(K,1), round(pH,2)
            except Exception:
                pass
    # Fallback heuristic mapping
    mapping = {
        "Sandy": (30,20,35,6.0),
        "Loamy": (50,30,40,6.5),
        "Clay": (60,35,50,7.2)
    }
    base = mapping.get(soil_type, mapping["Loamy"])
    N,P,K,pH = base
    if area_type.lower()=="dry":
        N *= 0.95; P*=0.98; K*=0.95
    if rainfall.lower()=="low":
        P*=0.95; K*=0.95
    if rainfall.lower()=="high":
        N*=1.02; K*=1.05
    return round(N,1),round(P,1),round(K,1),round(pH,2)

def soil_health_score(N,P,K,pH):
    score = 0
    score += max(0, 30 - abs(N-75)/2)
    score += max(0, 30 - abs(P-35)/1.5)
    score += max(0, 25 - abs(K-45)/2)
    score += max(0, 15 - abs(pH-6.8)*5)
    return int(max(0, min(100, score)))

def soil_health_tier(score):
    if score >= 85: return "BEST", "green"
    if score >= 70: return "GOOD", "lightgreen"
    if score >= 50: return "MEDIUM", "yellow"
    return "POOR", "red"

# ---------- UTIL: image classification helpers ----------
IMG_SIZE = (224,224)

def preprocess_for_model(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_with_model_safe(model, pil_img, labels=None):
    """Return (pred_label, confidence) or (None,0) if model missing"""
    if model is None:
        return None, 0.0
    try:
        x = preprocess_for_model(pil_img)
        preds = model.predict(x)
        if preds.ndim==1 or preds.shape[-1]==1:
            # regression or single output - not expected
            return None, float(preds[0])
        if labels:
            idx = int(np.argmax(preds))
            return labels[idx], float(preds[0,idx])
        else:
            # return class index and confidence
            idx = int(np.argmax(preds))
            return str(idx), float(preds[0,idx])
    except Exception as e:
        return None, 0.0

# ---------- PDF helper ----------
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    buf.seek(0)
    return buf

def pdf_report(user_inputs, analysis, charts_bytes):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0,10, "AgriVision AI - Soil Analysis Report", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0,8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)
    pdf.cell(0,8, "User Inputs:", ln=True)
    for k,v in user_inputs.items():
        pdf.cell(0,6, f"- {k}: {v}", ln=True)
    pdf.ln(4)
    pdf.cell(0,8, "Analysis Summary:", ln=True)
    for k,v in analysis.items():
        # safe multi-cell
        pdf.multi_cell(0,6, f"- {k}: {v}")
    pdf.ln(4)
    for i, img_bytes in enumerate(charts_bytes):
        try:
            img_bytes.name = getattr(img_bytes, "name", f"chart_{i}.png")
            pdf.image(img_bytes, w=160)
        except Exception as e:
            # if pdf.image fails, embed text mentioning missing chart
            pdf.cell(0,6, f"- (Could not embed chart {i}: {e})", ln=True)
    out = io.BytesIO()
    out.write(pdf.output(dest='S').encode('latin-1'))
    out.seek(0)
    return out

def get_table_download_link(pdf_bytes, filename="report.pdf"):
    b64 = base64.b64encode(pdf_bytes.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Report (PDF)</a>'
    return href

# ---------- PAGE: Home ----------
if page == "Home":
    st.header("Quick Soil Check & Recommendation")
    left, right = st.columns([2,1])
    with left:
        with st.form("home_form"):
            col1, col2 = st.columns(2)
            with col1:
                soil_type = st.selectbox("Soil Type", options=["Loamy","Sandy","Clay"])
                region = st.text_input("Region (optional)")
                area_type = st.selectbox("Area Type", options=["Humid","Dry"])
            with col2:
                rainfall = st.selectbox("Rainfall Pattern", options=["Low","Medium","High"])
                area_hectares = st.number_input("Area (hectares)", min_value=0.01, value=0.5, step=0.1)
            analyze = st.form_submit_button("Analyze My Soil")
        if analyze:
            N,P,K,pH = estimate_npk_ph_from_inputs(soil_type, region, area_type, rainfall)
            score = soil_health_score(N,P,K,pH)
            tier, color = soil_health_tier(score)
            # small chart (homepage small)
            fig_small = plt.figure(figsize=(3,2))
            ax = fig_small.add_subplot(111)
            ax.bar(["N","P","K"], [N,P,K])
            ax.set_ylim(0, max(100, N+10, P+10, K+10))
            ax.set_title("N-P-K (Est.)", fontsize=9)
            st.pyplot(fig_small)
            # score display with color
            st.markdown(f"<h3>Soil Health: <span style='color:{color}'>{tier} ({score}/100)</span></h3>", unsafe_allow_html=True)
            # bigger crop recommendation area
            st.markdown("### Recommended Crops (larger display)")
            # Try to use crop_df data intelligently
            recommended = []
            if isinstance(crop_df, pd.DataFrame):
                # try to use pH ranges if available
                pH_col = None
                for c in crop_df.columns:
                    if "ph" in c.lower():
                        pH_col = c; break
                crop_name_col = crop_df.columns[0]
                for _,row in crop_df.iterrows():
                    # try to interpret
                    try:
                        if pH_col:
                            val = str(row[pH_col])
                            if '-' in val:
                                low,high = val.split('-')
                                if float(low)<=pH<=float(high):
                                    recommended.append(str(row[crop_name_col]))
                            else:
                                recommended.append(str(row[crop_name_col]))
                        else:
                            recommended.append(str(row[crop_name_col]))
                    except:
                        continue
                # dedupe and show top 6
                recommended = list(dict.fromkeys(recommended))[:6]
            if not recommended:
                recommended = ["Wheat","Maize","Rice"]
            # show as larger tiles (plotly)
            fig_reco = go.Figure()
            fig_reco.add_trace(go.Bar(x=recommended, y=[1]*len(recommended), text=recommended, textposition="auto"))
            fig_reco.update_layout(height=300, showlegend=False, yaxis=dict(visible=False))
            st.plotly_chart(fig_reco, use_container_width=True)
            # fertilizer suggestion
            nutrient_list = {"N":N,"P":P,"K":K}
            lowest = min(nutrient_list, key=nutrient_list.get)
            if lowest == "N":
                suggested = "Urea (Nitrogen rich) or organic compost"
            elif lowest == "P":
                suggested = "DAP or Phosphorus-rich fertilizer"
            else:
                suggested = "MOP (Potassium rich) or organic compost"
            st.info(f"Lowest nutrient: {lowest} → Suggested: {suggested}")
            # Generate downloadable PDF button
            chart_bytes = [fig_to_bytes(fig_small)]
            user_inputs = {"Soil Type":soil_type,"Region":region,"Area Type":area_type,"Rainfall":rainfall,"Area (ha)":area_hectares}
            analysis = {"N":N,"P":P,"K":K,"pH":pH,"Score":f"{score}/100","Tier":tier,"Recommended Crops":", ".join(recommended),"Fertilizer Suggestion":suggested}
            pdf_bytes = pdf_report(user_inputs, analysis, chart_bytes)
            st.markdown(get_table_download_link(pdf_bytes, filename="AgriVision_report.pdf"), unsafe_allow_html=True)

# ---------- PAGE: Detection & Suggestion ----------
elif page == "Detection & Suggestion":
    st.header("Image Detection & Suggestion")
    st.markdown("Upload one image (soil / crop / fertilizer). The app will try to detect the type and give suggestions. If you have pre-trained models in `models/` folder they will be used.")
    uploaded = st.file_uploader("Upload Image (single)", type=["jpg","jpeg","png"])
    if uploaded:
        try:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded Image", width=350)
        except Exception as e:
            st.error(f"Could not open image: {e}")
            img = None
        detected_type = None
        det_conf = 0.0
        # 1) Try models
        if img is not None:
            # If you have dedicated class label files, load them here. For demo we'll assume models return label indexes.
            # We try all three models and pick the one with highest confidence.
            candidates = []
            s_label, s_conf = predict_with_model_safe(soil_model, img)
            if s_label: candidates.append(("Soil", s_label, s_conf))
            c_label, c_conf = predict_with_model_safe(crop_model, img)
            if c_label: candidates.append(("Crop", c_label, c_conf))
            f_label, f_conf = predict_with_model_safe(fert_model, img)
            if f_label: candidates.append(("Fertilizer", f_label, f_conf))
            # pick highest confidence
            if candidates:
                best = max(candidates, key=lambda x: x[2])
                detected_type, det_label, det_conf = best[0], best[1], best[2]
            else:
                # fallback: try filename heuristics
                name = uploaded.name.lower()
                if any(k in name for k in ["soil","sand","clay","loam"]):
                    detected_type, det_label, det_conf = "Soil", "Filename-based", 0.5
                elif any(k in name for k in ["crop","plant","leaf","wheat","rice","maize","soy"]):
                    detected_type, det_label, det_conf = "Crop", "Filename-based", 0.5
                elif any(k in name for k in ["fert","fertilizer","urea","dap","mop","compost","vermi"]):
                    detected_type, det_label, det_conf = "Fertilizer", "Filename-based", 0.5
                else:
                    detected_type, det_label, det_conf = "Unknown", "Unknown", 0.0
        st.markdown(f"**Detected Type:** {detected_type} (confidence {det_conf:.2f})")
        # Provide suggestions based on detected type
        if detected_type == "Soil":
            # ask minimal farmer-friendly inputs
            soil_type = st.selectbox("If known, soil type", ["Loamy","Sandy","Clay"])
            area_type = st.selectbox("Area Type", ["Humid","Dry"])
            rainfall = st.selectbox("Rainfall Pattern", ["Low","Medium","High"])
            N,P,K,pH = estimate_npk_ph_from_inputs(soil_type, region=None, area_type=area_type, rainfall=rainfall)
            score = soil_health_score(N,P,K,pH); tier,color = soil_health_tier(score)
            st.markdown(f"**Estimated N,P,K,pH:** {N}, {P}, {K}, {pH}")
            st.markdown(f"**Soil Health:** <span style='color:{color}'>{tier} ({score}/100)</span>", unsafe_allow_html=True)
            st.pyplot(plot_npk := plt.figure(figsize=(4,2))); ax = plot_npk.axes[0]; ax.bar(["N","P","K"], [N,P,K]); st.pyplot(plot_npk)
            # recommend crops
            recommended = []
            if isinstance(crop_df, pd.DataFrame):
                # try to find crop column
                crop_name_col = crop_df.columns[0]
                for _,row in crop_df.iterrows():
                    recommended.append(str(row[crop_name_col]))
                recommended = list(dict.fromkeys(recommended))[:6]
            st.write("Suggested crops:", recommended)
        elif detected_type == "Crop":
            # try to identify crop from model label or fallback to manual selection for detailed info
            crop_selected = None
            if isinstance(crop_df, pd.DataFrame):
                crop_name_col = crop_df.columns[0]
                # if det_label matches a crop name in dataset, pick it
                if det_label and isinstance(det_label, str):
                    for c in crop_df[crop_name_col].astype(str):
                        if det_label.lower() in c.lower():
                            crop_selected = c; break
                if not crop_selected:
                    crop_selected = st.selectbox("Select crop for detailed info", options=list(crop_df[crop_name_col].astype(str).unique()))
                info_row = crop_df[crop_df[crop_name_col].astype(str)==str(crop_selected)].iloc[0].to_dict()
                st.write(info_row)
            else:
                st.info("Crop dataset not available; show generic advice for common crops.")
        elif detected_type == "Fertilizer":
            # show fertilizer info from fert_df if available
            if isinstance(fert_df, pd.DataFrame):
                fert_name_col = fert_df.columns[0]
                # try match label
                match = None
                if det_label and isinstance(det_label, str):
                    for f in fert_df[fert_name_col].astype(str):
                        if det_label.lower() in f.lower():
                            match = f; break
                if not match:
                    match = st.selectbox("Select fertilizer for info", options=list(fert_df[fert_name_col].astype(str).unique()))
                finfo = fert_df[fert_df[fert_name_col].astype(str)==str(match)].iloc[0].to_dict()
                st.write(finfo)
            else:
                st.info("Fertilizer dataset not found; show general fertilizer safety tips.")
        else:
            st.warning("Could not confidently detect the image type. Try another image or select the type manually.")

        # FINAL combined recommendation logic
        st.markdown("---")
        st.markdown("### Final Recommendation (combined)")
        # We'll compute a simple combined recommendation:
        try:
            # If we have N,P,K and crop_selected or recommended list, do compatibility check
            if 'N' in locals() and isinstance(crop_df, pd.DataFrame):
                # pick top crop from crop_df that matches pH and nutrients roughly
                def crop_score(row):
                    s = 0
                    # check pH column if present
                    for c in crop_df.columns:
                        if "ph" in c.lower():
                            try:
                                val = str(row[c])
                                if '-' in val:
                                    low,high = val.split('-')
                                    if float(low) <= pH <= float(high): s+=1
                            except: pass
                    # check NPK ideal ranges if present by column names
                    for col_name, ideal_val in [('n','N'),('p','P'),('k','K')]:
                        for c in crop_df.columns:
                            if col_name in c.lower() and '-' in str(row[c]):
                                try:
                                    low,high = map(float, str(row[c]).split('-'))
                                    # add if p in ideal
                                    if low <= locals()[ideal_val] <= high:
                                        s+=1
                                except: pass
                    return s
                best_crops = []
                try:
                    # compute scores
                    scored = []
                    for _,r in crop_df.iterrows():
                        scored.append((str(r[crop_df.columns[0]]), crop_score(r)))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    best_crops = [c for c,sc in scored if sc>0][:3]
                except Exception:
                    best_crops = list(crop_df[crop_df.columns[0]].astype(str).unique())[:3]
                if not best_crops:
                    best_crops = list(crop_df[crop_df.columns[0]].astype(str).unique())[:3]
                st.write("Top matched crops:", ", ".join(best_crops))
            else:
                st.write("Not enough data to compute an advanced recommendation. Provide dataset files or fill fields.")
        except Exception as e:
            st.error(f"Final recommendation error: {e}")

# ---------- PAGE: ChatBot (Styled Gemini Integration - Streamlit 1.30+ Compatible) ----------
elif page == "ChatBot":
    import google.generativeai as genai
    from dotenv import load_dotenv
    import os
    import time
    import streamlit as st

    # Load environment variables
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        st.error("❌ Gemini API key not found. Please set it in your .env file.")
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("models/gemini-2.0-flash")

        # UI Header
        # st.markdown(
        #     "<h2 style='text-align:center; color:#2e7d32;'>🤖 Agri ChatBot (Powered by Google Gemini 🌾)</h2>",
        #     unsafe_allow_html=True
        # )
        st.markdown(
            "<p style='text-align:center; color:gray;'>Ask me anything about crops, soil, fertilizers, or modern farming!</p>",
            unsafe_allow_html=True
        )

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Chat container with scroll
        chat_container = st.container()
        with chat_container:
            for sender, msg in st.session_state.chat_history:
                if sender == "You":
                    st.markdown(
                        f"""
                        <div style='background-color:#DCF8C6; padding:10px 15px; border-radius:15px; margin:5px 0; max-width:75%; text-align:right; margin-left:auto; color:#000;'>
                        <b>🧑‍🌾 You:</b> {msg}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style='background-color:#F1F0F0; padding:10px 15px; border-radius:15px; margin:5px 0; max-width:75%; text-align:left; margin-right:auto; color:#000;'>
                        <b>🌾AgriVision : </b> {msg}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Input box
        user_input = st.text_input("💬 Type your message here:", key="user_input")

        # Send button
        if st.button("Send"):
            if user_input.strip():
                with st.spinner("💭 thinking..."):
                    try:
                        response = model.generate_content(user_input)
                        bot_reply = response.text.strip()

                        # Save chat history
                        st.session_state.chat_history.append(("You", user_input))
                        st.session_state.chat_history.append(("Gemini", bot_reply))

                        # ✅ New correct rerun for Streamlit >= 1.30
                        time.sleep(0.5)
                        st.rerun()

                    except Exception as e:
                        st.error(f"⚠️ Error: {e}")


# ---------- PAGE: Help Centre ----------
elif page == "Help Centre":
    st.header("Help Centre")
    st.markdown("About this app and how to use it.")
    with st.form("help"):
        name = st.text_input("Your name")
        email = st.text_input("Your email")
        msg = st.text_area("Describe issue / feedback")
        send = st.form_submit_button("Send")
    if send:
        try:
            if not OWNER_EMAIL:
                st.info("Email not configured. Edit OWNER_EMAIL in the script to enable email.")
            else:
                # send email
                em = EmailMessage()
                em["Subject"] = f"AgriVision Help: {name}"
                em["From"] = SMTP_USER
                em["To"] = OWNER_EMAIL
                em.set_content(f"From: {name} <{email}>\n\n{msg}")
                with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
                    smtp.starttls(); smtp.login(SMTP_USER, SMTP_PASS); smtp.send_message(em)
                st.success("Help request sent.")
        except Exception as e:
            st.error(f"Could not send email: {e}")

# ---------- PAGE: Downloads ----------
elif page == "Downloads":
    st.header("Downloads")
    st.markdown("CSV and model files available in the project directory (if present).")
    st.write("Soil CSV:", SOIL_CSV if os.path.exists(SOIL_CSV) else "Not found")
    st.write("Crop CSV:", CROP_CSV if os.path.exists(CROP_CSV) else "Not found")
    st.write("Fertilizer CSV:", FERT_CSV if os.path.exists(FERT_CSV) else "Not found")
    st.write("Models folder:", MODELS_DIR)
    st.write("Logo:", LOGO_PATH if os.path.exists(LOGO_PATH) else "Not found")




