import torch
import os
import pandas as pd
import warnings
from argparse import ArgumentParser
from datasets.crowd import Crowd
from models.vgg import vgg19

warnings.filterwarnings('ignore')

def parse_args():
    parser = ArgumentParser(description='Test')
    parser.add_argument('--data-dir', default='./data_processed', help='testing data directory')
    parser.add_argument('--save-dir', default='./result', help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    return parser.parse_args()

def test(args):
    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False, num_workers=0, pin_memory=False)
    model = vgg19().to(torch.device('cuda'))
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), torch.device('cuda')))
    counts = []

    for inputs, count, name in dataloader:
        inputs = inputs.to(torch.device('cuda'))
        with torch.no_grad():
            outputs = model(inputs)
            count = int(torch.sum(outputs).item())
            print(name, " Count: " ,count)
            counts.append(count)

    pred_data = {"Count":counts}
    df_pred = pd.DataFrame(pred_data)
    df_pred.index = df_pred.index + 1
    df_pred.to_csv('310581027_final_submission_result.csv', index_label='ID')

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    test(args)