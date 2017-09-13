import argparse
import pdb

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--c", type=open, action=LoadFromFile)
    parser.add_argument("-lr", "--lr", type=float, help="learning rate")
    parser.add_argument("-bsz", "--bsz", type=int, help="batch size")
    parser.add_argument("-epochs", "--epochs", type=int, help="number epochs")
    parser.add_argument("-lr_decay", "--lr_decay", type=float, help="learning rate decay rate")
    parser.add_argument("-lr_decay_epochs", "--lr_decay_epochs", type=int, help="learning rate decay epochs")
    parser.add_argument("-optimizer", "--optimizer", type=str, help="optimizer type")
    parser.add_argument("-momentum", "--momentum", type=float, help="momentum term")

    args = parser.parse_args()

    default_args = {'lr': 0.01,
                    'bsz': 32,
                    'epochs': 10,
                    'lr_decay': None,
                    'lr_decay_epochs': 10,
                    'optimizer': 'Momentum',
                    'momentum': 0.9
                    } 

    for field, value in default_args.items():
        if getattr(args, field) is None:
            setattr(args, field, value)

    return args
