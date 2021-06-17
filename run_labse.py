from model.layers_labse import Trainer, LaBSEEncoder
import torch


if __name__ == '__main__':
    trainer = Trainer(seed=37)
    trainer.train(0)
