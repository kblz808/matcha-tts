import tempfile
from pathlib import Path

import gradio as gr
import soundfile as sf
import torch

from matcha.cli import (
    get_device,
    load_matcha,
    load_vocoder,
    process_text,
    to_waveform,
)
from matcha.utils.utils import get_user_data_dir, plot_tensor

# Configuration - Replace with your model path
CUSTOM_CHECKPOINT_PATH = "/content/matcha-tts/logs/train/tuba/runs/2025-05-10_18-41-29/checkpoints/last.ckpt"
VOCODER_NAME = "hifigan_univ_v1"
LOCATION = Path(get_user_data_dir())

def VOCODER_LOC(x):
    return LOCATION / f"{x}"

# Setup device
device = get_device(None)

# Load custom model
print(f"[🍵] Loading custom model from {CUSTOM_CHECKPOINT_PATH}")
model = load_matcha("custom_model", CUSTOM_CHECKPOINT_PATH, device)
vocoder, denoiser = load_vocoder(VOCODER_NAME, VOCODER_LOC(VOCODER_NAME), device)

@torch.inference_mode()
def synthesize(text, n_timesteps=10, temperature=0.667, length_scale=0.95):
    """Synthesize speech from text using the custom model."""
    # Process text - directly from CLI
    text_processed = process_text(1, text, device)
    
    # Generate mel spectrogram - directly from CLI
    output = model.synthesise(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=None,  # Set to speaker ID tensor if needed
        length_scale=length_scale,
    )
    
    # Convert to waveform - directly from CLI
    output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, output["waveform"], 22050)
        return f.name

# Create Gradio interface
with gr.Blocks(title="Custom Matcha-TTS") as demo:
    gr.Markdown("# 🍵 Custom Matcha-TTS")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text to synthesize",
                placeholder="Enter text here...",
                lines=3
            )
            
            with gr.Row():
                n_timesteps = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Number of timesteps"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.667,
                    step=0.01,
                    label="Temperature"
                )
                length_scale = gr.Slider(
                    minimum=0.5,
                    maximum=1.5,
                    value=0.95,
                    step=0.01,
                    label="Length scale"
                )
            
            synthesize_btn = gr.Button("Synthesize")
        
        with gr.Column():
            audio_output = gr.Audio(label="Generated Speech")
    
    # Set up the synthesis function
    synthesize_btn.click(
        fn=synthesize,
        inputs=[text_input, n_timesteps, temperature, length_scale],
        outputs=audio_output
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)