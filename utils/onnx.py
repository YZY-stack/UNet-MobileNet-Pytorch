from UNet_MobileNet import UNet # 这个是Pytorch-Unet项目里面网络结构
import torch
import onnx

# gloabl variable
model_path = "unet_carvana_scale1_epoch5.pth"

if __name__ == "__main__":
    # input shape尽量选择能被2整除的输入大小
    dummy_input = torch.randn(1, 3, 640, 960, device="cuda")
    # print(dummy_input.shape)
    # [1] create network
    model = UNet(n_channels=3, num_classes=1)
    model_dict = model.state_dict()
    model = model.cuda()
    model.eval()
    print("create U-Net model finised ...")
    # [2] 加载权重
    state_dict = torch.load(model_path)
    # model.load_state_dict(state_dict)
    print("load weight to model finised ...")

    # 筛除不加载的层结构
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    # 更新当前网络的结构字典
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    # convert torch format to onnx
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model,
                      dummy_input,
                      "unet_deconv.onnx",
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names)
    print("convert torch format model to onnx ...")
    # [4] confirm the onnx file
    net = onnx.load("unet_deconv.onnx")
    # check that the IR is well formed
    onnx.checker.check_model(net)
    # print a human readable representation of the graph
    onnx.helper.printable_graph(net.graph)
