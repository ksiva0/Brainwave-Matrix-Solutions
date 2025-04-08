import streamlit as st  
from diffusers import StableDiffusionPipeline  
import torch  

# Load model and use GPU if available  
device = "cuda" if torch.cuda.is_available() else "cpu"  
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to(device)  

def generate_image(prompt):  
    # Generate image from text prompt with reduced steps for faster processing  
    image = pipe(prompt, num_inference_steps=25).images[0]  
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
