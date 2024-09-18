from torch.utils.mobile_optimizer import optimize_for_mobile
import torch
from model import CustomMobileNetV3
import onnx
from onnxsim import simplify
from torch.autograd import Variable
from option import get_args
opt = get_args()


model = CustomMobileNetV3()
model.load_state_dict(torch.load(f'{opt.checkpoints}model_best.pth', map_location='cpu')['model'])
model.eval()
print("Model loaded successfully.")


"""Save .pth format model"""
torch.save(model, f'{opt.checkpoints}/model.pth')


"""Save .ptl format model"""
example = torch.rand(1, 3, 64, 64)
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter(f'{opt.checkpoints}model.ptl')


"""Save .onnx format model"""
input_name = ['input']
output_name = ['output']
input = Variable(torch.randn(1, 3, opt.loadsize, opt.loadsize))
torch.onnx.export(model, input, f'{opt.checkpoints}model.onnx', input_names=input_name, output_names=output_name, verbose=True)
onnx.save(onnx.shape_inference.infer_shapes(onnx.load(f'{opt.checkpoints}model.onnx')), f'{opt.checkpoints}model.onnx')   # Perform shape judgment
# simplified model
model_onnx = onnx.load(f'{opt.checkpoints}model.onnx')
model_simplified, check = simplify(model_onnx)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simplified, f'{opt.checkpoints}model_simplified.onnx')

