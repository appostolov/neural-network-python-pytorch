import sys

from .manager import train, run

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "none"

    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        print("Command train or run missing!")


