import torch
from config import Config
from models.AlexNet import AlexNet, JigsawAlexNet

cfg = Config()

state_dict = torch.load(cfg.checkpoint_path)
for name, param in state_dict.items():
    print(name)

net = JigsawAlexNet()
# net.load(cfg.checkpoint_path)
for name, param in net.state_dict().items():
    print(name) 
