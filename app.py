import streamlit as st
from PIL import Image
import cv2
import numpy as np
from streamlit_image_comparison import image_comparison

st.set_page_config(
    page_title="Tamil Palm Leaf AI",
    page_icon="📜",
    layout="wide"
)

st.title("📜 Tamil Palm Leaf Manuscript Restoration")
st.caption("AI-Assisted Digital Heritage Preservation System")

st.write(
    "Restore degraded Tamil palm-leaf manuscripts and analyze "
    "possible damage using intelligent image processing."
)

uploaded_file = st.file_uploader(
    "📤 Upload a Palm Leaf Manuscript Image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # 1. Noise reduction
    denoised = cv2.medianBlur(gray, 3)

    # 2. Contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(denoised)

    # 3. Text restoration
    restored = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    st.markdown("## 🌓 Interactive Restoration Comparison")

    restored_rgb = cv2.cvtColor(
        restored,
        cv2.COLOR_GRAY2RGB
    )

    image_comparison(
        img1=image,
        img2=Image.fromarray(restored_rgb),
        label1="Original Degraded",
        label2="AI Restored",
        width=1000,
        starting_position=50,
        show_labels=True
    )

    # -----------------------------------------
    # AUTOMATIC DAMAGE ANALYSIS
    # -----------------------------------------

    st.markdown("## 🔍 Automatic Damage Analysis")

    # Dark pixel analysis
    dark_pixels = np.sum(gray < 80)
    total_pixels = gray.size

    dark_ratio = (dark_pixels / total_pixels) * 100

    # Bright pixel analysis
    bright_pixels = np.sum(gray > 220)
    bright_ratio = (bright_pixels / total_pixels) * 100

    # Contrast measurement
    contrast_value = np.std(gray)

    # Damage score
    damage_score = (
        (dark_ratio * 0.5) +
        ((100 - bright_ratio) * 0.2) +
        ((50 - min(contrast_value, 50)) * 0.3)
    )

    damage_score = max(0, min(damage_score, 100))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dark Region",
            f"{dark_ratio:.2f}%"
        )

    with col2:
        st.metric(
            "Contrast",
            f"{contrast_value:.2f}"
        )

    with col3:
        st.metric(
            "Damage Score",
            f"{damage_score:.2f}%"
        )

    if damage_score < 30:
        st.success("🟢 Low Damage — Manuscript condition is relatively good.")

    elif damage_score < 60:
        st.warning("🟡 Moderate Damage — Restoration is recommended.")

    else:
        st.error("🔴 High Damage — Significant degradation detected.")
    # -----------------------------------------
    # DAMAGE MAP VISUALIZATION
    # -----------------------------------------

    st.markdown("## 🗺️ Manuscript Damage Map")

    damage_map = cv2.GaussianBlur(gray, (9, 9), 0)

    _, damage_mask = cv2.threshold(
        damage_map,
        100,
        255,
        cv2.THRESH_BINARY_INV
    )

    damage_map_color = cv2.applyColorMap(
        damage_mask,
        cv2.COLORMAP_JET
    )

    damage_map_rgb = cv2.cvtColor(
        damage_map_color,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        damage_map_rgb,
        caption="Detected Potential Damage Regions",
        use_container_width=True
    )

    st.info(
        "🔎 Bright regions indicate areas requiring closer "
        "inspection during manuscript restoration."
    )

    # -----------------------------------------
    # RESTORATION PIPELINE
    # -----------------------------------------

    st.markdown("## 🔬 Restoration Pipeline")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "🔹 Noise Reduction\n\n"
            "Median Filter"
        )

    with col2:
        st.info(
            "🔹 Contrast Enhancement\n\n"
            "CLAHE"
        )

    with col3:
        st.info(
            "🔹 Text Restoration\n\n"
            "Adaptive Thresholding"
        )
    # -----------------------------------------
    # RESTORATION QUALITY DASHBOARD
    # -----------------------------------------

    st.markdown("## 📊 Restoration Quality Dashboard")

    original_contrast = np.std(gray)
    restored_contrast = np.std(restored)

    contrast_improvement = (
        ((restored_contrast - original_contrast)
         / (original_contrast + 1e-6)) * 100
    )

    original_edges = cv2.Canny(gray, 100, 200)
    restored_edges = cv2.Canny(restored, 100, 200)

    original_edge_density = np.mean(original_edges > 0) * 100
    restored_edge_density = np.mean(restored_edges > 0) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Original Contrast",
            f"{original_contrast:.2f}"
        )

    with col2:
        st.metric(
            "Restored Contrast",
            f"{restored_contrast:.2f}"
        )

    with col3:
        st.metric(
            "Contrast Change",
            f"{contrast_improvement:.2f}%"
        )

    st.write(
        f"**Original Edge Density:** "
        f"{original_edge_density:.2f}%"
    )

    st.write(
        f"**Restored Edge Density:** "
        f"{restored_edge_density:.2f}%"
    )

    st.progress(
        min(max(int(restored_edge_density), 0), 100)
    )

    st.caption(
        "The dashboard provides measurable image-quality "
        "indicators before and after restoration."
    )
    # -----------------------------------------
    # DOWNLOAD RESTORED IMAGE
    # -----------------------------------------

    st.markdown("## 💾 Download Restored Manuscript")

    restored_pil = Image.fromarray(restored)

    import io

    buffer = io.BytesIO()

    restored_pil.save(
        buffer,
        format="PNG"
    )

    st.download_button(
        label="⬇️ Download Restored Image",
        data=buffer.getvalue(),
        file_name="restored_palm_leaf.png",
        mime="image/png"
    )

    st.success("✅ Restoration and damage analysis completed successfully!")

else:

    st.info(
        "👆 Upload a palm-leaf manuscript image to begin."
    )

st.divider()

st.markdown(
    "### 🏛️ Digital Heritage Preservation"
)

st.write(
    "This system assists in digitally restoring degraded Tamil "
    "palm-leaf manuscripts while analyzing their visual degradation."
)