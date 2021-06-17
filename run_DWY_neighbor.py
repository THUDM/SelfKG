from model.layers_DWY_neighbor import Trainer


if __name__ == '__main__':
    trainer = Trainer(seed=37)
    print(trainer.model)
    trainer.train(0)