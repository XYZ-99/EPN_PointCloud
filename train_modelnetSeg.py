from SPConvNets.trainer_modelnetRotation import Trainer as TrainerR
from SPConvNets.trainer_modelnet import Trainer
from SPConvNets.options import opt


if __name__ == '__main__':
    opt.mode = 'train'
    # For choosing the Dataloader_ModelNet40Alignment. (-> see trainer_modelRotation.py)
    opt.model.flag = 'rotation'
    print("Performing a regression task on a segmentation-based backbone...")
    opt.model.model = 'seg_so3net'
    trainer = TrainerR(opt)
    trainer.train()
