
|                    | Description | Implementation |
|:------------------:|-------------|----------------|
| **Normalizer**     | Executes all the initial transformations over the initial input string. For example when you need to lowercase some text, maybe strip it, or even apply one of the common unicode normalization process, you will add a Normalizer. | Lowercase, Unicode (NFD, NFKD, NFC, NFKC), Bert, Strip, ... |
| **PreTokenizer**   | In charge of splitting the initial input string. That's the component that decides where and how to pre-segment the origin string. The simplest example would be like we saw before, to simply split on spaces. | ByteLevel, WhitespaceSplit, CharDelimiterSplit, Metaspace, ... |
| **Model**          | Handles all the sub-token discovery and generation, this part is trainable and really dependant of your input data. | WordLevel, BPE, WordPiece |
| **Post-Processor** | Provides advanced construction features to be compatible with some of the Transformers-based SoTA models. For instance, for BERT it would wrap the tokenized sentence around [CLS] and [SEP] tokens. | BertProcessor |
| **Decoder**        | In charge of mapping back a tokenized input to the original string. The decoder is usually chosen according to the PreTokenizer we used previously. | WordLevel, BPE, WordPiece, ... |
| **Trainer**        | Provides training capabilities to each model. | |



1. [**Normalizer**](https://huggingface.co/docs/tokenizers/python/latest/components.html#normalizers): Make it less random or “cleaner”. `tokenizers.normalizers`
   - `Lowercase`: Lowercasing all text. Most common
   - Stripping whitespace
   - Removing accented characters `StripAccents`
   - Unicode normalization (`NFD`, `NFKD`, `NFC`, `NFKC`)
2. [**Pre-Tokenization**](https://huggingface.co/docs/tokenizers/python/latest/components.html#pre-tokenizers): Split text into "megatokens" to feed the model `tokenizers.pre_tokenizers`
   - `Whitespace`: Most common
   - `Digits`
   - `ByteLevel`
   - `CharDelimiter`
   - `Metaspace`
3. [**Model**](https://huggingface.co/docs/tokenizers/python/latest/components.html#models):
   - `BPE`: Byte-Pair-Encoding starts with characters, then iteratively merges the most frequent together.
   - `WordPiece`: Similar to BPE. Starts with long words, then splits. Used in BERT.
   - `WordLevel`: Simply map words to IDs without anything fancy
   - `Unigram`: Other subword tokenization algorithm
4. [**Post-Processor**](https://huggingface.co/docs/tokenizers/python/latest/components.html#postprocessor): Adds spacial tokens like [CLS]
   - `TemplateProcessing`
   
   
## Use a pretrained tokenizer

```python
!pip install tokenizers

# Download pre-trained vocabulary file
!wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt

from tokenizers import BertWordPieceTokenizer
tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)

print(tokenizer.encode(text).tokens)
```


## Build a tokenizer from scratch

#### 1. Get a corpus
```bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip

# File from Peter Norving. 30.000 lines of raw text
wget https://norvig.com/big.txt
```


```python
from tokenizers                import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models         import BPE

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
```

Otro [ejemplo de kaggle](https://www.kaggle.com/funtowiczmo/hugging-face-tutorials-training-tokenizer)


```python
from tokenizers                import Tokenizer
from tokenizers.normalizers    import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.models         import BPE
from tokenizers.decoders       import ByteLevel as ByteLevelDecoder
from tokenizers.trainers       import BpeTrainer

##### CREATE

# First we create an empty Byte-Pair Encoding model (i.e. not trained model)
tokenizer = Tokenizer(BPE.empty())

# Then we enable lower-casing and unicode-normalization
# The Sequence normalizer allows us to combine multiple Normalizer that will be
# executed in order.
tokenizer.normalizer = Sequence([ NFKC(), Lowercase() ])

# Our tokenizer also needs a pre-tokenizer responsible for converting the input to a ByteLevel representation.
tokenizer.pre_tokenizer = ByteLevel()

# And finally, let's plug a decoder so we can recover from a tokenized input to the original one
tokenizer.decoder = ByteLevelDecoder()


#### TRAIN

# We initialize our trainer, giving him the details about the vocabulary we want to generate
trainer = BpeTrainer(vocab_size=25000, show_progress=True, initial_alphabet=ByteLevel.alphabet())
tokenizer.train(trainer, ["../input/big.txt"])

print("Trained vocab size: {}".format(tokenizer.get_vocab_size())) # Trained vocab size: 25000


#### SAVE

# You will see the generated files in the output.
tokenizer.model.save('.')  # ['./vocab.json', './merges.txt']

#### LOAD

# Let's tokenizer a simple input
tokenizer.model = BPE.from_files('vocab.json', 'merges.txt')


#### INFERENCE

encoding = tokenizer.encode("This is a simple input to be tokenized")
print("Encoded string: {}".format(encoding.tokens)) # ['Ġthis', 'Ġis', 'Ġa', 'Ġsimple', 'Ġin', 'put', 'Ġto', 'Ġbe', 'Ġtoken', 'ized']

decoded = tokenizer.decode(encoding.ids)
print("Decoded string: {}".format(decoded)) # Decoded string:  this is a simple input to be tokenized

```
