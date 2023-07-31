import torch.nn
model = torch.load("../result/best.pth", map_location=torch.device("cpu"))
x = torch.randn(1, 3, 224, 224)

torch.onnx.export(model, x, "../result/resnetntu.onnx", input_names=['input'], output_names=['output'], verbose=True)
