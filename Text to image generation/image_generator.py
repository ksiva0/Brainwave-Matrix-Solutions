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
        "runwayml/stable-diffusion-v1-5",  
        torch_dtype=dtype,  
    ).to(device)  

    # Use a better scheduler  
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)  
    
    # Enable memory-efficient attention if possible  
    try:  
        pipe.enable_xformers_memory_efficient_attention()  
    except ModuleNotFoundError:  
        print("xformers not installed, memory efficient attention disabled")  
    
    return pipe  

pipe = load_model()  

def generate_image(prompt):
    generator = torch.Generator(device=device).manual_seed(42)  # For reproducibility

    # Get max valid steps from the scheduler
    max_steps = len(pipe.scheduler.sigmas) - 1
    num_steps = min(20, max_steps)  # Use 20 or lower, depending on the model/scheduler

    image = pipe(
        prompt,
        num_inference_steps=num_steps,
        guidance_scale=7.5,
        generator=generator,
        height=512,
        width=512
    ).images[0]
    return image  

def main():  
    st.title("Text to Image Generator")  
    user_input = st.text_area("Enter a description:")  

    if st.button("Generate Image"):  
        if user_input:  
            with st.spinner("Generating image..."):  
                try:  
                    image = generate_image(user_input)  
                    st.image(image, caption="Generated Image", use_column_width=True)  
                except Exception as e:  
                    st.error(f"An error occurred while generating the image: {e}")  
        else:  
            st.warning("Please enter a description.")  

if __name__ == "__main__":  
    main()
