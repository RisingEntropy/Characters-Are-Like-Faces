import torch
import torch.onnx
from modules import MobileFaceNet

model = MobileFaceNet(embedding_size=64)
model.load_state_dict(torch.load("./saved_models/ocrnet_argued_19.pt"))
model.eval()

input = torch.rand(1, 3, 112, 112)
jit_model = torch.jit.trace(model, input)
torch.jit.save(jit_model, "OCRNet.pth")
torch.onnx.export(model, input, "OCRNet.onnx", export_params=True)