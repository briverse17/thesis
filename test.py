import argparse
from bwe import Embedder

def run(embedder):
    if embedder.data.raw_batches == []:
        embedder.data = embedder.data.load()

    if embedder.data.tokenized_batches == []:
        embedder.data = embedder.tokenize()

    embedder.data = embedder.bert_embed()

def main(params_file):
    embedder = Embedder(params_file)
    run(embedder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=str,
        default="./params.json",
        help="Path to the parameters (.json) file. Default to: ./params.json")
    args = parser.parse_args()
    main(args.params)