import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr

class_names = ['Audi', 'BMW', 'Tesla']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("car_brand_model.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image):
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
    return class_names[pred.item()]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Car Brand Classifier",
    description="Upload an image of a car and I'll tell you if it's a BMW, Audi, or Tesla."
)

if __name__ == "__main__":
    demo.launch()
