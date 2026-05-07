import os
import torch
import gradio as gr
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
import openai

# Set your OpenAI API key
openai.api_key = "AIzaSyDLaAEZQUPogis2O3uz0Ay2z83-O32IWcU"

# Load text generation model (GPT-4) for scriptwriting
def generate_manga_script(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Write a short manga script based on the prompt."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]

# Load Stable Diffusion model for manga-style illustration
diffusion_pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
diffusion_pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_manga_image(prompt):
    image = diffusion_pipeline(prompt).images[0]
    return image

# Character creation from an input photo (Placeholder: Implement ControlNet/DreamBooth)
def generate_character_image(photo):
    # Placeholder for AI-based character transformation
    return photo  # Returning original image for now

# Define the Gradio interface function
def manga_generator_ui(prompt, photo):
    script = generate_manga_script(prompt)
    manga_panel = generate_manga_image(prompt)
    character_image = generate_character_image(photo)
    return script, manga_panel, character_image

# Create the GUI using Gradio
iface = gr.Interface(
    fn=manga_generator_ui,
    inputs=[
        gr.Textbox(label="Enter Manga Prompt"),
        gr.Image(label="Upload Protagonist Photo")
    ],
    outputs=[
        gr.Textbox(label="Generated Script"),
        gr.Image(label="Generated Manga Panel"),
        gr.Image(label="Generated Character")
    ],
    title="AI Manga Generator",
    description="Generate a manga script and illustrations based on your input!"
)

# Run the application
if __name__ == "__main__":
    iface.launch()
