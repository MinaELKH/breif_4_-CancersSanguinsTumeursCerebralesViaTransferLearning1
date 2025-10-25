
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from ultralytics import YOLO
from pathlib import Path
import io

# ---------- CONFIG (adjust paths if needed) ----------
OUTPUT_ROOT = Path("./Data_Brain_filtered")
YoloWeights = OUTPUT_ROOT / "yolov8_best.pt"       
ClassifierWeights = Path("googlenet_best.pth")     
CLASS_NAMES = None                                 
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------

st.set_page_config(page_title="Brain Tumor: Classifier + YOLOv8", layout="wide")
st.title("Brain Tumor — Classifier (GoogLeNet) + YOLOv8 Detector")

def infer_class_names():
    # try dataset.yaml first
    yaml_path = OUTPUT_ROOT / "dataset.yaml"
    if yaml_path.exists():
        import yaml
        try:
            d = yaml.safe_load(open(yaml_path))
            names = d.get("names")
            if names:
                return names
        except Exception:
            pass
    # try metadata saved with classifier (if saved)
    meta_path = Path("training_metadata.json")
    if meta_path.exists():
        import json
        try:
            m = json.load(open(meta_path))
            if m.get("classes"):
                return m["classes"]
        except Exception:
            pass
    return ["class0", "class1", "class2", "class3"]

with st.sidebar:
    st.write("**Settings**")
    st.write(f"Device: {DEVICE}")
    show_classifier = st.checkbox("Run classifier (GoogLeNet)", value=True)
    show_detector = st.checkbox("Run YOLOv8 detector", value=True)
    conf_thr = st.slider("YOLO confidence threshold", min_value=0.01, max_value=0.99, value=0.25, step=0.01)
    iou_thr = st.slider("YOLO iou threshold (NMS)", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

# lazy-load models with caching
@st.cache_resource
def load_yolo(weights_path):
    if not Path(weights_path).exists():
        st.warning(f"YOLO weights not found at {weights_path}. Please run training cell first.")
        return None
    return YOLO(str(weights_path))

@st.cache_resource
def load_classifier(path_ckpt, device):
    # Build GoogLeNet same architecture as training and load state_dict. Minimal wrapper that expects same architecture.
    import torch.nn as nn
    from torchvision import models
    if not Path(path_ckpt).exists():
        st.warning(f"Classifier weights not found at {path_ckpt}. Skipping classifier.")
        return None, None
    # load class names if embedded in checkpoint
    try:
        ckpt = torch.load(str(path_ckpt), map_location=device)
        class_names = ckpt.get("class_names") or ckpt.get("classes") or None
    except Exception:
        ckpt = None
        class_names = None

    # instantiate GoogLeNet and replace fc structure (must match your training design)
    model = models.googlenet(pretrained=False, aux_logits=False, transform_input=False)
    in_features = model.fc.in_features
    # ensure classifier architecture matches what you trained; adjust if you changed head
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, len(class_names) if class_names else 4)  # fallback to 4
    )
    if ckpt is not None and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        try:
            model.load_state_dict(torch.load(str(path_ckpt)))
        except Exception as e:
            st.error(f"Failed to load classifier weights: {e}")
            return None, None

    model.to(device)
    model.eval()

    return model, class_names

# Load class names
CLASS_NAMES = infer_class_names()
st.sidebar.write("Classes:", CLASS_NAMES)

# Load models
yolo_model = None
classifier_model = None
if show_detector:
    yolo_model = load_yolo(YoloWeights)
if show_classifier:
    classifier_model, cls_names_from_ckpt = load_classifier(ClassifierWeights, DEVICE)
    if cls_names_from_ckpt:
        CLASS_NAMES = cls_names_from_ckpt

# transforms for classifier
clf_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# UI: upload or sample image selection
uploaded = st.file_uploader("Upload an image (brain MRI) to classify & detect", type=["jpg","jpeg","png","bmp","tiff"])
col1, col2 = st.columns([1,1])

def pil_to_bytes(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    # Run classifier
    if show_classifier and classifier_model is not None:
        input_t = clf_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = classifier_model(input_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            top_idx = int(probs.argmax())
            top_prob = float(probs[top_idx])
            class_label = CLASS_NAMES[top_idx] if 0 <= top_idx < len(CLASS_NAMES) else str(top_idx)
        st.markdown("### Classifier result")
        st.write(f"Predicted: **{class_label}** — confidence {top_prob:.3f}")
        # show full probs table
        import pandas as pd
        df = pd.DataFrame({"class": CLASS_NAMES, "prob": probs})
        st.dataframe(df.sort_values("prob", ascending=False).reset_index(drop=True))

    # Run YOLO detection
    annotated_img = None
    if show_detector and yolo_model is not None:
        st.markdown("### YOLOv8 detections")
        results = yolo_model.predict(source=img, conf=conf_thr, iou=iou_thr, device=DEVICE, verbose=False)
        r = results[0]
        # r.boxes contains xyxy coords, cls, conf
        # draw boxes on PIL image copy
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=16)
        except Exception:
            font = ImageFont.load_default()
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            st.write("No boxes detected (above confidence threshold).")
            annotated_img = annotated
        else:
            # boxes.xyxy, boxes.conf, boxes.cls
            xyxy = boxes.xyxy.cpu().numpy()   # (n,4)
            confs = boxes.conf.cpu().numpy()  # (n,)
            clsids = boxes.cls.cpu().numpy().astype(int)  # (n,)
            for (x0,y0,x1,y1), conf, cid in zip(xyxy, confs, clsids):
                label = (CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else f"class_{cid}")
                draw.rectangle([x0,y0,x1,y1], outline="red", width=3)
                text = f"{label} {conf:.2f}"
                tw, th = draw.textsize(text, font=font)
                draw.rectangle([x0, y0 - th - 4, x0 + tw + 4, y0], fill="red")
                draw.text((x0+2, y0 - th - 2), text, fill="white", font=font)
            annotated_img = annotated
            st.image(annotated_img, caption="YOLO annotated image", use_column_width=True)

    # Download annotated image
    if annotated_img is not None:
        buf = io.BytesIO()
        annotated_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("Download annotated image (PNG)", data=byte_im, file_name="annotated.png", mime="image/png")

else:
    st.info("Upload an image to run classification and detection. Example: select an MRI from Data_Brain_filtered images folder.")
    st.write("You can toggle classifier/detector in the sidebar and adjust YOLO thresholds.")
