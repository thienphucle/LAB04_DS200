from transforms.transforms import Transforms
from models.svm import SVM
from trainer import Trainer

if __name__ == "__main__":
    transforms = Transforms()
    model = SVM()
    trainer = Trainer(model, transforms)
    trainer.train()
