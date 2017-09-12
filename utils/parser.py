import argparse
import pdb

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)

def parse():
    parser = argparse.ArgumentParser()
    #parser.add_argument("-c", "--c", type=str, help="configuration file")
    parser.add_argument("-c", "--c", type=open, action=LoadFromFile)
    parser.add_argument("-lr", "--lr", type=float, help="learning rate")
    parser.add_argument("-bsz", "--bsz", type=int, help="batch size")
    parser.add_argument("-epochs", "--epochs", type=int, help="number epochs")
    parser.add_argument("-lr_decay", "--lr_decay", type=float, help="learning rate decay rate")
    parser.add_argument("-lr_decay_epochs", "--lr_decay_epochs", type=int, help="learning rate decay epochs")
    parser.add_argument("-optimizer", "--optimizer", type=str, help="optimizer type")

    args = parser.parse_args()
    return args
