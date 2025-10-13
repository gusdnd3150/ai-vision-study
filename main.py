
import torch
from src.Init import Init

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    # x = torch.tensor([1., 2.]).to(device)
    # print(f'x {x} {type(x)}')
    Init()

