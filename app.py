from fastapi import FastAPI
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import base64
import torchvision.transforms as T
import onnxruntime

app = FastAPI()

print("loading model ...")
model = onnxruntime.InferenceSession("./asset/palm_recognition.onnx")
print("Finish loading model!")

test_transform = T.Compose(
        [
            T.Resize([224, 224]),
            T.ToTensor(),
            T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )

class Item(BaseModel):
    base64_image: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/")
async def predict(item: Item):
    base64_image = item.base64_image
    # Decode image
    image = base64.b64decode(base64_image)
    # Convert to PIL Image
    image = Image.open(BytesIO(image))
    # Convert to PyTorch Tensor and add batch dimension
    image = test_transform(image)
    image = image.unsqueeze(0)
    ort_inputs = {model.get_inputs()[0].name: to_numpy(image)}
    ort_outs = model.run(None, ort_inputs)
    onnx_out = ort_outs[0]
    return {"prediction": onnx_out.tolist()}
