import gradio as gr

from model import CaptionModel

CHECKPOINT_PATH = "checkpoints/best_caption_model.pth"

model = CaptionModel(CHECKPOINT_PATH)


def generate_caption(image):
    if image is None:
        return "Please upload an image."
    return model.predict(image)


demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Textbox(label="Generated caption"),
    title="Image Captioning",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
