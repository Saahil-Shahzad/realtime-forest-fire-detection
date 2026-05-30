import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import timm
import os
import tempfile
import pandas as pd

# ============================================
# TIFF READING FUNCTION
# ============================================

def read_tiff(path):
    """Read TIFF file"""
    try:
        import tifffile
        arr = tifffile.imread(path)
    except Exception:
        import imageio.v2 as imageio
        arr = imageio.imread(path)
    arr = np.asarray(arr).astype(np.float32)
    return arr

# ============================================
# TIFF TO 3-CHANNEL CONVERSION
# ============================================

def to_3channel_uint8_pil(arr, percent_clip=(1, 99)):
    """Convert thermal TIFF to RGB PIL image"""
    
    # If 3D (RGB), take first channel
    if arr.ndim == 3:
        arr = arr[..., 0]

    # Handle NaNs/Infs (replace with median)
    arr = np.nan_to_num(arr, nan=np.nanmedian(arr), posinf=np.nanmedian(arr), neginf=np.nanmedian(arr))

    # Percent clipping
    low_p, high_p = percent_clip
    vmin = np.percentile(arr, low_p)
    vmax = np.percentile(arr, high_p)

    if vmax <= vmin:
        vmax = arr.max() if arr.max() != vmin else vmin + 1.0

    # Normalize to 0-255
    arr_clipped = np.clip(arr, vmin, vmax)
    arr_norm = (arr_clipped - vmin) / (vmax - vmin)
    arr_255 = (arr_norm * 255.0).round().astype(np.uint8)

    # Make 3-channel (grayscale to RGB)
    if arr_255.ndim == 2:
        arr_rgb = np.stack([arr_255, arr_255, arr_255], axis=2)
    else:
        if arr_255.shape[2] >= 3:
            arr_rgb = arr_255[..., :3]
        else:
            arr_rgb = np.concatenate([arr_255] * 3, axis=2)

    pil = Image.fromarray(arr_rgb)
    return pil

# ============================================
# PROCESS UPLOADED FILE
# ============================================

def process_uploaded_file(uploaded_file):
    """Process uploaded TIFF file"""
    
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tiff') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    try:
        # Read TIFF
        arr = read_tiff(temp_path)
        
        # Convert to 3-channel RGB
        pil_image = to_3channel_uint8_pil(arr, percent_clip=(1, 99))
        
        return pil_image
    
    finally:
        # Clean up temp file
        os.unlink(temp_path)

# ============================================
# IMAGE TRANSFORM
# ============================================

def get_transform(target_size=(224, 224)):
    """Get the transform"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    return transform

# ============================================
# MODEL DEFINITION
# ============================================

class OptimizedThermalChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        reduced = max(channels // reduction_ratio, 8)

        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced), nn.ReLU(),
            nn.Linear(reduced, reduced), nn.ReLU(),
            nn.Linear(reduced, channels),
        )
        self.bn = nn.BatchNorm1d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        gap = self.gap(x).view(b, c)
        weights = self.mlp(gap)
        weights = self.bn(weights)
        weights = self.sigmoid(weights).view(b, c, 1, 1)
        return x * weights

class OTFAN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv4_conv_small_035', pretrained=False, num_classes=0)
        test_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            test_output = self.backbone(test_input)
        feature_channels = test_output.shape[1]
        self.otca = OptimizedThermalChannelAttention(feature_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_channels, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        otca_features = self.otca(features.unsqueeze(-1).unsqueeze(-1))
        otca_features = otca_features.squeeze(-1).squeeze(-1)
        return self.classifier(otca_features)

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="OTFAN · Fire Detection",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    /* ── Global Reset & Base ── */
    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    /* ── Kill the white Streamlit top header bar ── */
    [data-testid="stHeader"],
    header[data-testid="stHeader"],
    .stAppHeader,
    #stDecoration {
        background-color: #0b0d11 !important;
        background: #0b0d11 !important;
        border-bottom: 1px solid #1a1e2a !important;
    }

    /* Deploy button and top-right controls */
    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"] {
        background-color: #0b0d11 !important;
    }

    /* ── Main app background ── */
    .stApp {
        background-color: #0b0d11;
        background-image:
            radial-gradient(ellipse 80% 40% at 50% -10%, rgba(255,60,30,0.08) 0%, transparent 70%),
            repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(255,255,255,0.02) 39px, rgba(255,255,255,0.02) 40px),
            repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(255,255,255,0.02) 39px, rgba(255,255,255,0.02) 40px);
        color: #e2e4e9;
    }

    /* ── Remove default top padding pushed in by header ── */
    .main .block-container {
        padding-top: 2rem !important;
    }

    /* ── Header ── */
    h1 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2.4rem !important;
        letter-spacing: -0.03em !important;
        color: #ffffff !important;
    }
    h2, h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        color: #c8cad0 !important;
        letter-spacing: -0.01em !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #10131a !important;
        border-right: 1px solid #1e2130 !important;
    }
    [data-testid="stSidebar"] * {
        font-family: 'Space Mono', monospace !important;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-family: 'Syne', sans-serif !important;
        color: #ff5533 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
    }

    /* Metric cards in sidebar */
    [data-testid="stMetric"] {
        background: #181c27;
        border: 1px solid #252a3a;
        border-radius: 6px;
        padding: 10px 14px !important;
        margin-bottom: 8px;
    }
    [data-testid="stMetric"] label {
        font-size: 0.65rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: #6b7280 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace !important;
        font-size: 1.4rem !important;
        color: #ff5533 !important;
    }

    /* ── File uploader — full dark theme ── */
    [data-testid="stFileUploader"] {
        background: #0e1118 !important;
        border-radius: 8px !important;
    }
    /* The inner drop zone / browse area */
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {
        background: #0e1118 !important;
        border: 1px dashed #2a3048 !important;
        border-radius: 8px !important;
        color: #4a5270 !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #ff5533 !important;
    }
    /* Upload button inside the dropzone */
    [data-testid="stFileUploaderDropzone"] button {
        background: #1a1f30 !important;
        color: #8892aa !important;
        border: 1px solid #2a3048 !important;
        border-radius: 4px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.75rem !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #222840 !important;
        border-color: #ff5533 !important;
        color: #ff5533 !important;
    }
    /* The small text label inside uploader */
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] p {
        color: #3a4260 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.72rem !important;
    }
    /* File name chip after upload */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
        background: #131720 !important;
        border: 1px solid #252a3a !important;
        border-radius: 4px !important;
        color: #8892aa !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.75rem !important;
    }

    /* ── Detect button ── */
    .stButton > button {
        background: linear-gradient(135deg, #cc2200 0%, #ff4422 100%) !important;
        color: #ffffff !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 0 24px rgba(255,68,34,0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #e02800 0%, #ff5533 100%) !important;
        box-shadow: 0 0 40px rgba(255,68,34,0.45) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Result cards ── */
    .fire-card {
        background: linear-gradient(145deg, #1a0a08 0%, #230e0a 100%);
        padding: 32px 28px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #7a1a0f;
        box-shadow: 0 0 60px rgba(200,30,10,0.15), inset 0 1px 0 rgba(255,100,80,0.1);
        position: relative;
        overflow: hidden;
    }
    .fire-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #ff4422, transparent);
    }
    .fire-card h1 {
        font-size: 2rem !important;
        color: #ff5533 !important;
        margin-bottom: 4px !important;
        letter-spacing: -0.02em !important;
    }
    .fire-card h2 {
        font-family: 'Space Mono', monospace !important;
        font-size: 1.1rem !important;
        color: #ff9980 !important;
        font-weight: 400 !important;
    }
    .fire-card p {
        font-family: 'Space Mono', monospace;
        color: #8a7070;
        font-size: 0.8rem;
        margin: 4px 0;
    }

    .no-fire-card {
        background: linear-gradient(145deg, #08140e 0%, #0c1a11 100%);
        padding: 32px 28px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #1a5c30;
        box-shadow: 0 0 60px rgba(10,160,60,0.1), inset 0 1px 0 rgba(50,220,120,0.08);
        position: relative;
        overflow: hidden;
    }
    .no-fire-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #22cc66, transparent);
    }
    .no-fire-card h1 {
        font-size: 2rem !important;
        color: #22cc66 !important;
        margin-bottom: 4px !important;
        letter-spacing: -0.02em !important;
    }
    .no-fire-card h2 {
        font-family: 'Space Mono', monospace !important;
        font-size: 1.1rem !important;
        color: #80ddaa !important;
        font-weight: 400 !important;
    }
    .no-fire-card p {
        font-family: 'Space Mono', monospace;
        color: #608070;
        font-size: 0.8rem;
        margin: 4px 0;
    }

    /* ── Alerts / info boxes ── */
    [data-testid="stAlert"] {
        border-radius: 6px !important;
        border-left-width: 3px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* ── Info box ── */
    .stInfo {
        background: #111620 !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: #111620 !important;
        border: 1px solid #1e2333 !important;
        border-radius: 6px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* ── Divider ── */
    hr {
        border-color: #1e2333 !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Caption / footer ── */
    .stMarkdown small, caption, .stCaption {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.7rem !important;
        color: #3a4055 !important;
        letter-spacing: 0.04em !important;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] {
        color: #ff5533 !important;
    }

    /* ── Bar chart axis labels ── */
    .vega-embed * {
        font-family: 'Space Mono', monospace !important;
    }

    /* ── Image container ── */
    [data-testid="stImage"] {
        border: 1px solid #1e2333;
        border-radius: 6px;
        overflow: hidden;
    }

    /* ── Section headers ── */
    .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #ff5533;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, #2a1a16, transparent);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model(model_path='otfan_fire_detection.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = OTFAN(num_classes=2)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            st.success(f"✅ Model loaded from {model_path}")
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.warning(f"⚠️ Model file '{model_path}' not found!")
    
    model = model.to(device)
    model.eval()
    return model, device

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict(model, device, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence, probabilities.cpu().numpy()[0]

# ============================================
# MAIN APP
# ============================================

def main():
    st.markdown("""
    <div style="padding: 8px 0 24px 0;">
        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.18em;text-transform:uppercase;color:#ff5533;margin-bottom:10px;">
            ◈ THERMAL IMAGING SYSTEM · v2.4
        </div>
        <h1 style="font-family:'Syne',sans-serif;font-weight:800;font-size:2.6rem;color:#fff;letter-spacing:-0.04em;margin:0;line-height:1.1;">
            OTFAN<br><span style="color:#ff5533;">Wildfire Detection</span>
        </h1>
        <p style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#4a5068;margin-top:12px;letter-spacing:0.03em;">
            Optimized Thermal Channel Attention Network · FLAME-3 Dataset · 5-Fold Cross Validation
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#1e2333;margin-bottom:28px;">', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding:20px 0 8px 0;">
            <div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:0.16em;text-transform:uppercase;color:#ff5533;">
                ◈ System Status
            </div>
            <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#22cc66;margin-top:6px;">
                ● Online · GPU Ready
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div style="font-family:Syne,sans-serif;font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:#ff5533;margin-bottom:10px;">Model Performance</div>', unsafe_allow_html=True)
        st.metric("Accuracy", "96.8%")
        st.metric("Precision", "96.9%")
        st.metric("Recall", "98.5%")
        st.metric("F1-Score", "97.7%")
        st.metric("Best Fold", "97.96%")
        
        st.markdown("---")
        st.markdown('<div style="font-family:Syne,sans-serif;font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:#ff5533;margin-bottom:10px;">Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#6b7a99;line-height:2;background:#10131a;padding:14px;border-radius:6px;border:1px solid #1e2333;">
            <span style="color:#8892aa;">Backbone</span><br>MobileNetV4 Conv Small 035<br>
            <span style="color:#8892aa;">Attention</span><br>OTCA Module<br>
            <span style="color:#8892aa;">Parameters</span><br>842,146<br>
            <span style="color:#8892aa;">Inference</span><br>203 FPS<br>
            <span style="color:#8892aa;">Input</span><br>224 × 224 px
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div style="font-family:Syne,sans-serif;font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:#ff5533;margin-bottom:10px;">Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#6b7a99;background:#10131a;padding:14px;border-radius:6px;border:1px solid #1e2333;line-height:2;">
            <span style="color:#8892aa;">Source</span><br>FLAME-3 Dataset<br>
            <span style="color:#ff5533;">622</span> fire · <span style="color:#22cc66;">116</span> non-fire<br>
            <span style="color:#8892aa;">Modality</span><br>Thermal Infrared
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-label">Input · Thermal Image</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a TIFF image...", 
            type=['tiff', 'tif', 'jpg', 'jpeg', 'png'],
            help="Upload thermal images (TIFF format recommended)"
        )
        
        if uploaded_file is not None:
            # Check if it's TIFF
            if uploaded_file.type in ['image/tiff', 'image/tif'] or uploaded_file.name.endswith(('.tiff', '.tif')):
                with st.spinner("Processing thermal TIFF image..."):
                    image = process_uploaded_file(uploaded_file)
                    st.success("✅ TIFF image processed")
            else:
                image = Image.open(uploaded_file).convert('RGB')
                st.info("📷 Regular image loaded")
            
            st.image(image, caption="Processed Thermal Frame", use_container_width=True)
            
            if st.button("🔍 Detect Fire", type="primary", use_container_width=True):
                with st.spinner("Analyzing with OTFAN model..."):
                    model, device = load_model()
                    transform = get_transform(target_size=(224, 224))
                    image_tensor = transform(image).unsqueeze(0)
                    pred_class, confidence, probs = predict(model, device, image_tensor)
                    
                    st.session_state['prediction'] = pred_class
                    st.session_state['confidence'] = confidence
                    st.session_state['probs'] = probs
                    st.rerun()
        else:
            st.markdown("""
            <div style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#3a4260;text-align:center;padding:40px 20px;border:1px dashed #1e2333;border-radius:8px;margin-top:8px;">
                ↑ Upload a TIFF thermal image<br>
                <span style="font-size:0.65rem;color:#2a3048;">Supported: .tiff · .tif · .jpg · .png</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-label">Output · Detection Result</div>', unsafe_allow_html=True)
        
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            confidence = st.session_state['confidence']
            probs = st.session_state['probs']
            
            if pred == 1:
                st.markdown(f"""
                <div class="fire-card">
                    <h1>⬥ FIRE DETECTED</h1>
                    <h2>Confidence: {confidence:.2%}</h2>
                    <p>Fire Probability &nbsp;·&nbsp; {probs[1]:.4f}</p>
                    <p>No-Fire Probability &nbsp;·&nbsp; {probs[0]:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
                st.error("⚠ HIGH ALERT: Fire signature detected in thermal frame")
            else:
                st.markdown(f"""
                <div class="no-fire-card">
                    <h1>⬥ SCENE CLEAR</h1>
                    <h2>Confidence: {confidence:.2%}</h2>
                    <p>No-Fire Probability &nbsp;·&nbsp; {probs[0]:.4f}</p>
                    <p>Fire Probability &nbsp;·&nbsp; {probs[1]:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
                st.success("✓ No fire signature detected in thermal frame")
            
            # FIXED: Use proper DataFrame for bar chart
            st.markdown('<div class="section-label" style="margin-top:20px;">Probability Distribution</div>', unsafe_allow_html=True)
            chart_data = pd.DataFrame({
                'Class': ['No Fire', 'Fire'],
                'Probability': [probs[0], probs[1]]
            })
            st.bar_chart(chart_data.set_index('Class'))
            
            with st.expander("Raw Output"):
                st.write(f"**Prediction:** {'Fire' if pred == 1 else 'No Fire'}")
                st.write(f"**Confidence:** {confidence:.4f}")
                st.write(f"**Class 0 (No Fire):** {probs[0]:.4f}")
                st.write(f"**Class 1 (Fire):** {probs[1]:.4f}")
        else:
            st.markdown("""
            <div style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#2a3048;text-align:center;padding:60px 20px;border:1px dashed #1a2030;border-radius:8px;margin-top:8px;">
                Awaiting input<br>
                <span style="font-size:0.65rem;color:#1e2535;">Run detection to see results here</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<hr style="border-color:#1e2333;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#2a3048;letter-spacing:0.06em;text-align:center;padding:8px 0;">
        OTFAN · FLAME-3 · 5-Fold CV · Best Accuracy 97.96% · MobileNetV4 Backbone
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()