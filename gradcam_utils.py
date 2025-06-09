"""
简单版 Grad‑CAM（用于 CNN / MobileNetV2）
调用: heatmap = get_gradcam(model, input_tensor, class_idx)
"""
import torch, torch.nn.functional as F

def _find_last_conv_with_name(model):
    # 对于 MobileNetV2，我们需要找到最后一个卷积层
    last_conv_name = None
    last_conv_module = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and name.startswith('features'):
            last_conv_name = name
            last_conv_module = module
    return last_conv_name, last_conv_module

def get_gradcam(model, x, class_idx=None):
    model.eval()
    # conv = _find_last_conv(model) # 旧的调用方式
    target_conv_name, target_conv_module = _find_last_conv_with_name(model)

    if target_conv_module is None:
        print("Error: Grad-CAM_utils - Could not find a suitable convolutional layer in model.features.")
        return torch.zeros(x.shape[2:]).numpy()
    print(f"Info: Grad-CAM_utils - Using target convolutional layer: {target_conv_name} ({target_conv_module})")

    feats, grads = [], []

    def forward_hook(_, __, output): 
        feats.append(output.detach())
        # print(f"Debug: Grad-CAM_utils - Forward hook triggered. feats length: {len(feats)}")
    def backward_hook(_, grad_in, grad_out): 
        grads.append(grad_out[0].detach())
        # print(f"Debug: Grad-CAM_utils - Backward hook triggered. grads length: {len(grads)}")

    fh = target_conv_module.register_forward_hook(forward_hook)
    bh = target_conv_module.register_backward_hook(backward_hook)

    out = model(x)
    if class_idx is None:
        class_idx = out.argmax(dim=1)
        print(f"Info: Grad-CAM_utils - Predicted class_idx: {class_idx.item()}")
    else:
        print(f"Info: Grad-CAM_utils - Using provided class_idx: {class_idx}")
        
    one_hot = torch.zeros_like(out)
    one_hot[0, class_idx] = 1
    out.backward(gradient=one_hot, retain_graph=True) # Added retain_graph=True just in case, though might not be needed for single call

    fh.remove(); bh.remove()
    
    if len(grads) == 0 or len(feats) == 0:
        print("Error: Grad-CAM_utils - Grads or Feats list is empty. Hooks might not have captured data.")
        return torch.zeros(x.shape[2:]).numpy()
    
    print(f"Debug: Grad-CAM_utils - feats[0] shape: {feats[0].shape}, min: {feats[0].min():.4f}, max: {feats[0].max():.4f}, mean: {feats[0].mean():.4f}")
    print(f"Debug: Grad-CAM_utils - grads[0] shape: {grads[0].shape}, min: {grads[0].min():.4f}, max: {grads[0].max():.4f}, mean: {grads[0].mean():.4f}")
        
    weights = grads[0].mean((2, 3), keepdim=True)
    print(f"Debug: Grad-CAM_utils - weights shape: {weights.shape}, min: {weights.min():.4f}, max: {weights.max():.4f}, mean: {weights.mean():.4f}")
    
    cam_before_relu = (weights * feats[0]).sum(1, keepdim=True)
    print(f"Debug: Grad-CAM_utils - cam_before_relu shape: {cam_before_relu.shape}, min: {cam_before_relu.min():.4f}, max: {cam_before_relu.max():.4f}, mean: {cam_before_relu.mean():.4f}")
    
    cam = F.relu(cam_before_relu)
    print(f"Debug: Grad-CAM_utils - cam_after_relu shape: {cam.shape}, min: {cam.min():.4f}, max: {cam.max():.4f}, mean: {cam.mean():.4f}")

    if cam.max() == 0 and cam.min() == 0:
        print("Warning: Grad-CAM_utils - cam after ReLU is all zeros. Heatmap will be blank/blue.")

    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    # 归一化
    cam_min = cam.min()
    cam_max = cam.max()
    print(f"Debug: Grad-CAM_utils - cam_after_interpolate shape: {cam.shape}, min: {cam_min:.4f}, max: {cam_max:.4f}")
    
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8) # More standard normalization
    # cam = (cam - cam.min()) / (cam.max() + 1e-8) # Your original normalization

    print(f"Debug: Grad-CAM_utils - cam_after_normalization shape: {cam.shape}, min: {cam.min():.4f}, max: {cam.max():.4f}, mean: {cam.mean():.4f}")
    return cam[0, 0].cpu().numpy()
