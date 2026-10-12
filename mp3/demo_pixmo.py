import torch
import sys
import tqdm
import logging
from absl import app, flags
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import yaml
from model import PixMoModel, Trainer, inference
from datasets import get_pixmo_data

FLAGS = flags.FLAGS
flags.DEFINE_string('exp_name', 'pixmo_demo',
                    'The experiment with corresponding hyperparameters to run. See config.yaml')
flags.DEFINE_string('output_dir', 'runs_pixmo', 'Output Directory')
flags.DEFINE_string('data_dir', './pixmo_data', 'Directory with pixmo data')

def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s: %(levelname)s %(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logging.getLogger().handlers = []
    if len(logging.getLogger().handlers) == 0: 
        logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)

def logger(tag, value, global_step):
    if tag == '':
       logging.info('')
    else:
       logging.info(f'  {tag:>15s} [{global_step:07d}]: {value:5f}')

class SummaryWriterWithPrinting(SummaryWriter):
    def add_scalar(self, tag, value, global_step): 
        super(SummaryWriterWithPrinting, self).add_scalar(tag, value, global_step)
        logger(tag, value, global_step)

def get_config(exp_name):
    dir_name = f'{FLAGS.output_dir}'

    # add/modify hyperparameters of your class in config.yaml
    encoder_registry = {
        'PixMoModel': PixMoModel,
    }
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[exp_name]

    net_class = encoder_registry[config['net_class']]
    num_classes = config['num_classes']
    batch_size = config['batch_size']
    # Everything under `model:` is passed to the model constructor as keyword
    # arguments (e.g. patch_tokens, num_layers, num_heads, feature_dim, dropout).
    model_kwargs = dict(config.get('model', {}) or {})
    # Optimizer / schedule settings. Absent keys default to wd=0.0,
    # momentum=0.9 and no scheduler; nothing in the block is silently ignored.
    # float() guards against YAML reading '3e-4' as a string (write 3.0e-4).
    train_kwargs = dict(
        optimizer=config['optimizer'],
        lr=float(config['lr']),
        epochs=config['epochs'],
        wd=float(config.get('wd', 0.0)),
        momentum=float(config.get('momentum', 0.9)),
        scheduler=config.get('scheduler', None),
    )
    known = {'net_class', 'num_classes', 'batch_size', 'model', 'optimizer', 'lr',
             'epochs', 'wd', 'momentum', 'scheduler'}
    unknown = sorted(set(config) - known)
    if unknown:
        raise KeyError(f"Unknown keys in config block '{exp_name}': {unknown}. "
                       f"Model settings go under 'model:'.")

    return net_class, (num_classes, model_kwargs), dir_name, train_kwargs, batch_size


def main(_):
    setup_logging()
    torch.set_num_threads(4)
    torch.manual_seed(0)

    print(f"Running: {FLAGS.exp_name}")

    net_class, (num_classes, model_kwargs), dir_name, train_kwargs, batch_size = \
        get_config(FLAGS.exp_name)
    lr, epochs = train_kwargs['lr'], train_kwargs['epochs']
    print(f"Config: model={model_kwargs} train={train_kwargs} batch_size={batch_size}")

    train_data = get_pixmo_data(FLAGS.data_dir,'train')
    val_data = get_pixmo_data(FLAGS.data_dir,'val')
    test_data = get_pixmo_data(FLAGS.data_dir,'test')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')  # Mac users
    else:
        device = torch.device('cpu')

    # Worker processes only help on the GPU cluster. On macOS they are spawned
    # (not forked) and each one would re-pickle the whole in-memory dataset.
    num_workers = 4 if device.type == 'cuda' else 0
    # drop_last keeps BatchNorm in the classifier from seeing a batch of size 1
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=num_workers, persistent_workers=num_workers > 0)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, persistent_workers=num_workers > 0)

    tmp_file_name = dir_name + '/best_model.pth'

    writer = SummaryWriterWithPrinting(f'{dir_name}', flush_secs=10)

    model = net_class(num_classes, **model_kwargs)
    model.to(device)

    trainer = Trainer(model, train_dataloader, val_dataloader, writer,
                      device=device, **train_kwargs)

    best_val_acc, best_epoch = trainer.train(model_file_name=tmp_file_name)
    print(f"lr: {lr:0.7f}, best_val_acc: {best_val_acc}, best_epoch: {best_epoch}")

    print("Training complete--------------------")

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model.load_state_dict(torch.load(f'{dir_name}/best_model.pth', weights_only=True))
    inference(test_dataloader, model, device, result_path=dir_name + '/test_pixmo.txt')


if __name__ == '__main__':
    app.run(main)
