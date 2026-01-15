import torch
from .transpath import TransPathModel
from .lit_model import LitTransPath
from .datamodule import GridDataModule
from .data_prep import GridDataset
from matplotlib import pyplot as plt

model_path = './weights/model_20241207-224531'
dataset_dir = ''
output = f'./eval_data/eval'
mode = 'cf'  # 'f', 'h', 'cf', 'nastar
indices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

eval_model = TransPathModel()
eval_model.load_state_dict(torch.load(model_path, weights_only=True))
eval_model.eval()

test_data = GridDataset(
        path=f'{dataset_dir}/test',
        mode=mode
    )[indices]

map_design, start, goal, gt_hmap = test_data
inputs = torch.cat([map_design, start + goal], dim=1) if mode in ('f', 'nastar') else torch.cat([map_design, goal], dim=1)
predictions = (eval_model(inputs) + 1) / 2
for i in range(len(indices)):
    plt.imshow(gt_hmap[i, 0], cmap='gray')
    plt.show()
    plt.imshow(predictions[i, 0].cpu().detach().numpy(), cmap='gray')
    plt.show()
my_test_data = GridDataset(
        path=output,
        mode=mode
    )[indices]

map_design, start, goal, gt_hmap = my_test_data
inputs = torch.cat([map_design, start + goal], dim=1) if mode in ('f', 'nastar') else torch.cat([map_design, goal], dim=1)
predictions = (eval_model(inputs) + 1) / 2

for i in range(len(indices)):
    plt.imshow(1 - (map_design[i, 0] + 5 * start[i, 0] + 5 * goal[i, 0]), cmap='gray')
    plt.show()
    plt.imshow(predictions[i, 0].cpu().detach().numpy(), cmap='gray')
    plt.show()