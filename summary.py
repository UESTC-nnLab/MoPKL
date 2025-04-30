import torch
from thop import clever_format, profile
from nets.MoPKL import MoPKL 

if __name__ == "__main__":
    input_shape = [512, 512]
    num_classes = 1
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model   = MoPKL(num_classes=num_classes, num_frame=2).to(device)
    model.eval()  
    dummy_input = torch.randn(1, 3, 2, input_shape[0], input_shape[1]).to(device)
    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")

    print(f"Total GFLOPs: {flops}")
    print(f"Total Params: {params}")
