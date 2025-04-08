import streamlit as st  
import requests  
from PIL import Image  
from io import BytesIO  

# Constants  
API_URL = "https://api.openai.com/v1/images/generations" 
API_KEY = st.secrets["api"]["key"] 

def main():  
    st.title("Text to Image Generation")  
    st.subheader("Generate images from textual descriptions")  

    # Input text from the user  
    user_input = st.text_area("Enter a description:", "")  
    
    if st.button("Generate Image"):  
        if user_input:  
            with st.spinner("Generating image..."):  
                image_url = generate_image(user_input)  
                if image_url:  
                    display_image(image_url)  
                else:  
                    st.error("Failed to generate image.")  
        else:  
            st.warning("Please enter a description.")  

def generate_image(description):  
    # Call the API to generate an image  
    payload = {"prompt": description}  
    headers = {  
        "Authorization": f"Bearer {API_KEY}",  
        "Content-Type": "application/json"  
    }  
    
    response = requests.post(API_URL, json=payload, headers=headers)  
    
    if response.status_code == 200:  
        # Assuming the response returns a URL of the image  
        return response.json().get("data")[0].get("url")  # Adjust based on your API response  
    else:  
        st.error(f"Error: {response.status_code} - {response.text}")  
        return None  

def display_image(image_url):  
    image_response = requests.get(image_url)  
    
    if image_response.status_code == 200:  
        image = Image.open(BytesIO(image_response.content))  
        st.image(image, caption="Generated Image", use_column_width=True)  
    else:  
        st.error("Could not retrieve the image.")  

if __name__ == "__main__":  
    main()  
