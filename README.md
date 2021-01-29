
# BERT WORD-EMBEDDING HELPER

This project generates BERT word-embedding from the input.
Input sentences must be preprocessed, word-segmented (required for Vietnamese) and stored in .txt format (one sentence per line).

## Command line interface
### Positional arguments:

- `input`: Path to the input file (.txt)

### Optional arguments:
- `-h/--help`: show help
- `-m/--model`:   HuggingFace model identifier, or
                Path to local directory containing neccesary files.
                Default: phobert-base
- `-l/--layers`:  Indicate the layer(s) to generate word-embedding.
                Cases:
                - 'all': output is generated using all layers with indicated `strategy`
                - 1 number (up to 12 values - 12 layers): output is generated
                using specified layer with indicated `strategy`
                - more than 1 number: output is generatednusing the specified layers with indicated strategy`
                The script automatically re-arrange the layers
                to ascending layers. To disable this behavior,
                add flag `--disable-ascending`.
                Default: 12