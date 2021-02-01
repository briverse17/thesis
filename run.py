import argparse
from bwe import Embedder

def main(params_file):
    embedder = Embedder(params_file)
    embedder.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=str,
        default="./params.json",
        help="Path to the parameters (.json) file. Default to: ./params.json")
    args = parser.parse_args()
    main(args.params)