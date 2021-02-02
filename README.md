
# BERT WORD-EMBEDDER

This project generates BERT word-embeddings from the input.
Input sentences must be preprocessed, word-segmented (required for Vietnamese) and stored in .txt format (one sentence per line).

## Usage

### Supported models (and associated tokenizers)
- `vinai/phobert-base`
- `vinai/phobert-large`
- `bert-base-cased`
- `bert-base-uncased`
- `bert-base-multilingual-cased`
- `bert-base-multilingual-uncased`

### Layers

List of distinct integers from 0 to 11

### Strategies

- `mean`: takes the average of layers' embeddings
- `concat`: concatenates layers' embeddings (beware of very long vectors!)
- `sum`: takes the sum of layers' embeddings (beware of huge values!)

### Class `Embedder`

*Tokenizes and embeds input batches*

**Attributes**
- `model`: a PyTorch model loaded by `HuggingFace Transformers`, default to `vinai/phobert-base`
- `toker`: default to `vinai.phobert-base` and must be a `fast` tokenizer. Be careful if you use a tokenizer that is not associated with the model.
- `toker_kwargs`: `dict`, default to `{"max_length": None, "padding": "max_length", "truncation": True, "return_tensors": "pt"}`
- `data`: an instance of [`Data`](#-class-data)
- `layers`: `list`, list of layers to get embeddings of, default to `[11]`
- `strategy`: `str`, strategy to combine layers' embeddings, default to `mean`
- `dump`: `bool`, save embeddings to `.npy` files or not, default to `True`

**Methods**
- `read_params()`: reads params specified in `.json` files and assigns to attributes
- `tokenize()`: tokenizes `Embedder.data.raw_batches` into `Embedder.data.tokenized_batches`
- `bert_embed()`: embeds `Embedder.data.tokenized` into `Embedder.data.embedded_batches`
- `combine_layers(embeddings: torch.Tensor, strategy: str)`: static method, combine layers' embeddings using the specified strategy

### Class `Data`

*Corresponds to load text input from `.txt` files and dump `np.array` to `.npy` files*

**Attributes**

- `input`: `str`, path to a single `.txt` file or a folder of `.txt` files
- `output`: `str`, path to a folder to dump `.npy` file(s) to
- `raw_batches`: `List[List[str]]`, each sublist contains sentences (a batch) read from a `.txt` file (initalized as `[]`, updated by `Data.load()`)
- `tokenized_batches`: `List[Dict["input_ids": torch.Tensor, "attention_mask": torch.Tensor]` each dictionary contains `input_ids` and `attention_mask` of a batch (initalized as `[]`, updated by `Embedder.tokenize()`)
- `embedded_batches`: `List[torch.Tensor]`, each `torch.Tensor` contains embeddings of every sentences in a batch (initalized as `[]`, updated by `Embedder.bert_embert()`)

**Methods**

- `load()`: read text input from path specified by `Data.input`
- `dump()`: save `Data.embedded_batches` as `np.array` to path specified by `Data.output`