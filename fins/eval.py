import os
import argparse
import torch
from torch.utils.data import DataLoader

from data.process_data import load_rir_dataset, load_speech_dataset
from trainer import Trainer
from dataloader import ReverbDataset
from model import FilteredNoiseShaper
from utils.utils import load_config


def main(args):
    # load config
    config_path = "fins/config.yaml"
    config = load_config(config_path)
    print(config)

    config.train.params.batch_size = 1

    if torch.cuda.is_available():
        args.device = torch.device("cuda", 0)
        torch.cuda.set_device(args.device)
    else:
        args.device = "cpu"
    print(args.device)

    # load dataset
    dataset = ReverbDataset(
        ["/share/home/tj13070/yueyinqiu/Ricbe--RirBlindEstimation/.gitignored/eval/rir.wav"], 
        ["/share/home/tj13070/yueyinqiu/Ricbe--RirBlindEstimation/.gitignored/eval/speech.wav"],
        config.dataset.params, 
        use_noise=False)

    valid_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        num_workers=config.train.params.num_workers,
    )

    # load model
    model = FilteredNoiseShaper(config.model.params)

    # run trainer
    trainer = Trainer(model, valid_dataloader, valid_dataloader, config.train.params, config.eval.params, args)

    trainer.plot_eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default='cpu')
    parser.add_argument("--save_name", type=str, default="m")
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    args = parser.parse_args()

    args.checkpoint_path = "/share/home/tj13070/yueyinqiu/fins/checkpoints/m-241229-215028/epoch-35.pt"

    main(args)
