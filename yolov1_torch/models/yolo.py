import torch 
import yaml
def make_layer(name,layer,kernel_size,channel_out,stride,padding): 
    assert layer in ['Conv','MaxPool','Flatten','Dense','Reshape'], 'Invalid layer type'
    if layer=='Conv':
        return Conv(name,kernel_size,channel_out)

class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Yolo(torch.nn.Module):
    
    def __init__(self,cfg='yolov1.yaml',nc=80): 
        super(Yolo, self).__init__()
        self.cfg = cfg
        if not isinstance(cfg,dict):
            with open(cfg,'r') as f:
                self.cfg = yaml.safe_load(f)
        if hasattr(self.cfg,'nc'):
            self.nc = self.cfg.get('nc')
        else:
            self.nc = nc
        self.m = []
        model = self.cfg.get('model')
        for index,layer_cfg in enumerate(model): 
            if index ==0: 
                channel_input = 3 
            name,layer_type,kernel_size,channel_out,stride,padding = layer_cfg
            # print(layer_cfg)
            if layer_type =='Conv':
                layer = torch.nn.Conv2d(in_channels = channel_input,\
                                        out_channels = channel_out,\
                                        kernel_size = kernel_size,\
                                        stride = stride,\
                                        padding = padding)
            elif layer_type == 'MaxPool': 
                layer = torch.nn.MaxPool2d(kernel_size=kernel_size,\
                                            stride = stride,\
                                            padding=padding)
            elif layer_type == 'Dense':
                print(channel_input,channel_out)
                layer = torch.nn.Linear(in_features = channel_input,\
                                        out_features = channel_out)
            elif layer_type == 'Flatten': 
                layer = torch.nn.Flatten()
            else: 
                layer = Reshape(7,7,30)
            self.m.append(layer)
            channel_input = channel_out
        self.m = torch.nn.ModuleList(self.m)

    def forward(self,x):
        for index,l in enumerate(self.m):
            x = l(x)
            # print(x.shape)
        return x

            

if __name__ =='__main__': 
    model = Yolo(cfg = '/u01/Intern/chinhdv/implement_yolo/yolov1_torch/models/hub/yolov1.yaml')
    x = torch.rand(1,3,448,448)
    out = model(x)
    print(out.shape)
            


