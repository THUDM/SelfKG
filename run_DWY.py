from model.layers_DWY import Trainer


if __name__ == '__main__':
    trainer = Trainer(seed=37)
    trainer.train(0)
