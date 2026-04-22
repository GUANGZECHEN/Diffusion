import sys
import os

sys.path.append(os.path.abspath("../src"))

from train import train

if __name__ == "__main__":
    train()
