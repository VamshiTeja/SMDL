import argparse
from models import *


def main():
    parser = argparse.ArgumentParser(description="SMILe: SubModular Incremental Learning")
    parser.add_argument("--repeat-rounds", default=5, type=int, help="The number of rounds the whole experiment needs"
                                                                     " to be performed.")
    parser.add_argument("--class-per-episodes", default=100, type=int, help="Number of classes introduced per episode.")
    parser.add_argument("--epochs", default=70, type=int, help="Number of epochs each episode needs to be trained.")
    parser.add_argument("--batch-size", default=64, type=int, help="Size of each batch of datapoints for SGD.")
    parser.add_argument("--learning-rate", default=0.1, type=float, help="Initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum parameter")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay")

    args = parser.parse_args()
    print args

if __name__ == "__main__":
    main()
