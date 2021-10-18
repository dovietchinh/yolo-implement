import torch 
import logging
import platform
LOGGER = logging.getLogger(__name__)

def select_device(devices=''): 
    s = f'YOLOv1 💥💥💥💥💥  torch {torch.__version__}'
    device = str(devices).strip().lower().replace('cuda:','')
    devices = device.split(',') if device else '0'
    cpu = device=='cpu'
    cuda = not cpu and torch.cuda.is_available()
    if cuda: 
        for i,d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else: 
        s += 'CPU\n'
    LOGGER.info(s.encode().decode('ascii','ignore') if platform.system()=='Windows' else s)
    return torch.device('cuda:0' if cuda else 'cpu')
    


if __name__ =='__main__':
    device = select_device(devices='cuda:0')
    print('-------------------------')
    print(LOGGER) 
    3,4,5,6,9,10->17,36