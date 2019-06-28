import os, sys
import argparse
from trainer import Trainer


def main(args):
    trainer = Trainer(model=args.r)
    trainer.train()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', choices=['Mymodel', 'GPR'])
    args = parser.parse_args()
    main(args)