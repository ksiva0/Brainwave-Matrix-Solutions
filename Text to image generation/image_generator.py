import streamlit as st
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch

# Determine device and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Load model with improved settings
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",  # Use a more refined model
        torch_dtype=dtype,
    ).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config) # change scheduler
    pipe.enable_xformers_memory_efficient_attention() #enable xformers
    return pipe

pipe = load_model()

def generate_image(prompt):
    # Generate image from text prompt with optimized parameters
    generator = torch.Generator(device=device).manual_seed(42)  # For reproducibility
    image = pipe(
        prompt,
        num_inference_steps=30,  # Slightly increase steps for better quality
        guidance_scale=7.5,      # Adjust guidance scale for better prompt adherence
        generator=generator,
        height=512, # set size
        width=512 # set size
    ).images[0]
    return image

def main():
    st.title("Text to Image Generator")
    user_input = st.text_area("Enter a description:")

    if st.button("Generate Image"):
        if user_input:
            with st.spinner("Generating image..."):
                image = generate_image(user_input)
                st.image(image, caption="Generated Image", use_column_width=True)
        else:
            st.warning("Please enter a description.")

if __name__ == "__main__":
    main()
