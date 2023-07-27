import torch.nn
model = torch.load("../results/best0.pt", map_location=torch.device("cpu"))
x = torch.randn(1, 3, 224, 224)

torch.onnx.export(model, x, "../results/resnetntu.onnx", input_names=['input'], output_names=['output'], verbose=True)
