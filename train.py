import wandb
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

import holocron
from trainer import ClassificationTrainer
from cnn_model import cnn_model



def target_transform(target):

  delta = 0.9

  target = delta*target + (1-delta)*(1-target)

  target = torch.tensor(target, dtype=torch.float32)

  return target.unsqueeze(dim = 0)


def build_dataset(config):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    size = config['image_size']
    if config['crop'] == 'random_crop':
        train_transforms = transforms.Compose([
              transforms.RandomCrop(size=size),
              transforms.RandomRotation(degrees=5),
              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              normalize
          ])

        val_transforms = transforms.Compose([
            transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
            normalize
        ])
    elif config['crop'] == 'full_card':

        train_transforms = transforms.Compose([
              transforms.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
              transforms.RandomRotation(degrees=5),
              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              normalize
          ])

        val_transforms = transforms.Compose([
            transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
            normalize
        ])


    dsTrain = ImageFolder('data/ds_matrix/train/', train_transforms, target_transform=target_transform)
    dsVal = ImageFolder('data/ds_matrix/test/', val_transforms, target_transform=target_transform)
    #dsTest = ImageFolder('data/WildFire/test/', val_transforms, target_transform=target_transform)
   
    train_loader = DataLoader(dsTrain, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(dsVal, batch_size=config['batch_size'], shuffle=True)
    #test_loader = DataLoader(dsTest, batch_size=config['batch_size'], shuffle=True)

    return train_loader, val_loader


def build_network(config):
    model_cut = -2
    num_classes=1
    lin_features=512
    dropout_prob=0.5
    bn_final=False
    concat_pool=True

    model_arch = config['model_arch']
    print(model_arch)
    base_model = holocron.models.__dict__[model_arch](False)

    if model_arch[:6]=='rexnet':
        nb_features = base_model.head[1].in_features

    elif model_arch[:6]=='resnet':
        nb_features = base_model.head.in_features

    else:
        nb_features=1024 #darknet


    model = cnn_model(base_model, model_cut, nb_features, num_classes,
                    lin_features, dropout_prob, bn_final=bn_final, concat_pool=concat_pool)


    return model


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        train_loader, val_loader = build_dataset(config)
        model = build_network(config)
        lr = config['lr']
        wd = config['wd']

        # Loss function
        criterion = nn.BCEWithLogitsLoss()

        # Create the contiguous parameters.
        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = holocron.optim.RAdam(model_params, lr, betas=(0.95, 0.99), eps=1e-6, weight_decay=wd)


        #Trainer
        trainer = ClassificationTrainer(model, train_loader, val_loader, criterion, optimizer, 0, output_file=config['checkpoint'], configwb=True)

        trainer.fit_n_epochs(config['epochs'], config['lr'], config['freeze'])
