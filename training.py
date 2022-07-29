import argparse
import torch
import pytorch_lightning as pl

from torch_geometric.datasets.zinc import ZINC
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.data import DataLoader
from .models import PSNLightning

if __name__ == '__main__':
    # Print the version of PyTorch
    print('Torch version:', torch.__version__)

    parser = argparse.ArgumentParser(description='Molecule classification')

    # Directories and dataset
    parser.add_argument('--root', default='../dataset', type=str, help='Path to save directory, default to "dataset"')
    parser.add_argument('--runs_dir', default='runs', type=str)
    parser.add_argument('--results_dir', default='results', type=str)

    # Training hyperparameters
    parser.add_argument('--epochs', default=500, type=int, help='Num. training epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--patience', default=50, type=int, help='The patience for early stopping.')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay parameter.')

    # Polynomial filter settings
    parser.add_argument('--filter_blocks', default=16, type=int, help='Number of filters (blocks) to use.')
    parser.add_argument('--layers_per_filter', default=2, type=int, help='Number of layers for each filter.')
    parser.add_argument('--embedding_size', default=128, type=int, help='The size for the embedding in the layers.')

    # Regularization
    parser.add_argument('--dropout_mlp', action='store_true', help='Whether to use dropout.')
    parser.add_argument('--dropout_mlp_probability', default=0.1, type=float, help='The probability of dropout.')

    # Others
    parser.add_argument('--number', default=0, type=int, help='Run number.')
    parser.add_argument('--gpu', default=0, type=int, help='GPU index. (-1 for CPU)')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Num. workers for Dataloader.')
    parser.add_argument('--name', default='psn', type=str, help='Name')

    args = parser.parse_args()

    # Set the device
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu') if args.gpu != -1 else 'cpu'

    # Set the seed
    pl.trainer.seed_everything(0)

    # Logger
    logger = TensorBoardLogger(save_dir=f'{args.runs_dir}/',
                               name=f'{args.name}_{args.filter_blocks}_{args.layers_per_filter}'
                               f'layers_{args.number}')

    # Checkpoint
    checkpoint_callback_loss = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'{args.results_dir}/{args.name}_{args.filter_blocks}_{args.layers_per_filter}',
        filename=f'{args.number}' + '{epoch:03d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )

    # Early stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min'
    )

    callbacks = [checkpoint_callback_loss, early_stopping_callback]

    # Trainer
    trainer = Trainer(gpus=[args.gpu],
                      logger=logger,
                      max_epochs=args.epochs,
                      callbacks=callbacks)

    # Download the dataset
    train_dataset = ZINC(f'{args.root}/zinc', split='train', subset=True)
    val_dataset = ZINC(f'{args.root}/zinc', split='val', subset=True)

    # Initiate the model
    model = PSNLightning(filter_blocks=args.filter_blocks,
                         layers_per_filter=args.layers_per_filter,
                         input_size=28,
                         classes=1,
                         lr=args.lr,
                         embedding_size=args.embedding_size,
                         dropout_mlp=args.dropout_mlp,
                         dropout_mlp_probability=args.dropout_mlp_probability,
                         weight_decay=args.weight_decay).to(device)

    # Training
    trainer.fit(model,
                train_dataloader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers),
                val_dataloaders=DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers))
