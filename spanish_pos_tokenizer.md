Subword tokenization español a nivel morfologico

Un etiquetador gramatical es un programa responsable de realizar POStagging

 estable (adjetivo,
sufijo “-able”) → estabilizar (verbo, sufijo “-izar”) → estabilizaci´on (sustantivo, sufijo “-aci´on”) → desestabilizaci´on (sustantivo, prefijo “des-”,
sufijo “-aci´on”).


############## TAGGER


EAGLES: Por ejemplo: a la palabra “bonita” le corresponde la etiqueta “AQ0FS0”
que representa un adjetivo (la “A” inicial), calificativo (“Q”), femenino (la
“F”) y singular (la “S”),


- NLTK (python)
pip install nltk
python -m nltk.downloader cess_esp
from nltk.corpus import cess_esp

- FreeLing (C++)
  - INPUT: archivo de texto.
  - OUTPUT: tagset EAGLES

- TreeTagger (C++)
  - OUTPUT: tagset propio. (mucho menos descriptivo que el EAGLES)

- MaltParser (Java)
  - OUTPUT: tagset EAGLES?

- Stanford Parser (Java)
  - OUTPUT: tagset EAGLES?



2 embeddings

Lemma embeddings
POS tag embedding



IDEA 2:

Transofrmer Attention solo en el arbol sintatico (parsing)




