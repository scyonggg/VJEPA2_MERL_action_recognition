import argparse
import gradio as gr
import torch
import cv2
import numpy as np
from pathlib import Path
import yaml
import tempfile
import threading
import time

# --- Import your model code ---
from evals.video_classification_frozen.models import init_module
from src.models.attentive_pooler import AttentiveClassifier

# --- MERL Shopping Dataset action label mapping ---
ACTION_LABELS = {
    0: "Reach To Shelf",
    1: "Retract From Shelf",
    2: "Hand In Shelf",
    3: "Inspect Product",
    4: "Inspect Shelf",
}

# --- Video processing and model inference ---
def process_video(video_path, encoder, classifier, device, frames_per_clip=16, resolution=256):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        print("Warning: FPS is 0 or NaN, setting to default 25.")
        fps = 25  # fallback default
    # Get original video resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"fps: {fps}, width: {width}, height: {height}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        out_path = tmpfile.name
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Failed to open VideoWriter for {out_path}")

    frames = []
    results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize and normalize frame for model
        frame_resized = cv2.resize(frame, (resolution, resolution))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frames.append(frame_tensor)
        # Process in clips
        if len(frames) == frames_per_clip:
            clip = torch.stack(frames)  # [T, C, H, W]
            clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, C, T, H, W]
            with torch.no_grad():
                features = encoder([[clip]])[0]  # Now each view is [B, C, T, H, W]
                logits = classifier(features)
                pred = logits.argmax(dim=1).item()
                label = ACTION_LABELS.get(pred, str(pred))
            # Overlay label on all frames in the clip
            for f in frames:
                f_disp = (f.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                f_disp = cv2.cvtColor(f_disp, cv2.COLOR_RGB2BGR)
                # Resize back to original video resolution for output
                f_disp = cv2.resize(f_disp, (width, height))
                cv2.putText(f_disp, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                out.write(f_disp)
            frames = []
    # Write remaining frames (if any)
    for f in frames:
        f_disp = (f.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        f_disp = cv2.cvtColor(f_disp, cv2.COLOR_RGB2BGR)
        f_disp = cv2.resize(f_disp, (width, height))
        cv2.putText(f_disp, "-", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        out.write(f_disp)
    cap.release()
    out.release()
    return out_path

# --- Gradio interface function ---
def gradio_infer(video, encoder, classifier, device):
    processed_path = process_video(video, encoder, classifier, device)
    # Schedule temp file cleanup after a longer delay
    def cleanup(path):
        time.sleep(60)  # Wait 60 seconds to ensure Gradio has read the file
        try:
            Path(path).unlink()
        except Exception:
            pass
    threading.Thread(target=cleanup, args=(processed_path,), daemon=True).start()
    print(f"Processed video saved at: {processed_path}")  # Debug print
    return processed_path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Main function ---
def main():
    parser = argparse.ArgumentParser(description="V-JEPA2 Gradio Video Action Recognition Demo")
    parser.add_argument('--config', type=str, required=True, help='Path to merl.yaml config')
    parser.add_argument('--encoder_ckpt', type=str, required=True, help='Path to encoder pretrain checkpoint (vjepa2_vitl.pt)')
    parser.add_argument('--classifier_ckpt', type=str, required=True, help='Path to best_val.pt checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on (cuda, cpu, or mps)')
    args = parser.parse_args()

    # --- Device selection logic (supports cuda, cpu, mps) ---
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            print("Warning: MPS device requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    # --- Load config ---
    config = load_config(args.config)
    model_kwargs = config['model_kwargs']['pretrain_kwargs']
    wrapper_kwargs = config['model_kwargs'].get('wrapper_kwargs', {})
    module_name = config['model_kwargs']['module_name']
    frames_per_clip = config['experiment']['data']['frames_per_clip']
    resolution = config['experiment']['data']['resolution']
    num_heads = config['experiment']['classifier']['num_heads']
    depth = config['experiment']['classifier']['num_probe_blocks']
    num_classes = config['experiment']['data']['num_classes']

    # --- Load encoder ---
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=args.encoder_ckpt,
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        device=device,
    )
    encoder.eval()
    encoder.to(device)

    # --- Load classifier ---
    classifier_ckpt = torch.load(args.classifier_ckpt, map_location='cpu')
    state_dict = classifier_ckpt['classifier']
    # Remove 'module.' prefix if present
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=num_heads,
        depth=depth,
        num_classes=num_classes
    )
    classifier.load_state_dict(state_dict, strict=True)
    classifier.eval()
    classifier.to(device)

    # --- Gradio UI ---
    def gradio_wrapper(video):
        return gradio_infer(video, encoder, classifier, device)

    demo = gr.Interface(
        fn=gradio_wrapper,
        inputs=gr.Video(label="Upload Video"),
        outputs=gr.Video(label="Processed Video with Action Labels"),
        title="V-JEPA2 Video Action Recognition Demo",
        description="Upload a video or use your webcam. The model will recognize and localize actions in real-time.",
        allow_flagging="never",
        live=False,
    )
    demo.launch()

if __name__ == "__main__":
    main() 