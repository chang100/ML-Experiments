import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--lr", type=float, help="learning rate")
    parser.add_argument("-bsz", "--bsz", type=int, help="batch size")
    parser.add_argument("-epochs", "--epochs", type=int, help="number epochs")
    parser.add_argument("-lr_decay", "--lr_decay", type=float, help="learning rate decay rate")
    parser.add_argument("-lr_decay_epochs", "--lr_decay_epochs", type=int, help="learning rate decay epochs")
    parser.add_argument("-optimizer", "--optimizer", type=str, help="optimizer type")

    args = parser.parse_args()
    return args
