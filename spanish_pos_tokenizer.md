# Tokenizador morfologico en Español

Un etiquetador gramatical es un programa responsable de realizar PoS tagging.

```
Text --> Tokens --> Lema    --> Lema Embeding --> Sum --> Final word embedding
                |-> PoS tag --> PoS Embeding  --/
```

### Ejemplos

|                   | LEMA      | PoS TAG                                      |
|-------------------|-----------|----------------------------------------------|
| socialista        | social    | ADJETIVO (sufijo "ista")                     |
| socialmente       | social    | ADVERVIO (sufijo "mente")                    |
| estable           | estable   | ADJETIVO (sufijo "able")                     |
| estabilizar       | estable   | VERBO       (sufijo "-izar")                 |
| estabilización    | estable   | SUSTANTIVO  (sufijo "ción")                  |
| desestabilización | estable   | SUSTANTIVO, (prefijo "des-", sufijo "ción")  |
| inadecuadamente   | adecuar   | ADVERVIO (prefijo "in-", sufijo "mente")     |


### Tagsets

EAGLES: Por ejemplo: a la palabra "bonita" le corresponde la etiqueta "AQ0FS0"
que representa un adjetivo (la "A" inicial), calificativo ("Q"), femenino (la
"F") y singular (la "S"),


# Software

|                                                        | Lang   | Tagset  |
|--------------------------------------------------------|--------|---------|
| NLTK                                                   | Python |         |
| [FreeLing](http://nlp.lsi.upc.edu/freeling/index.php/) | C++    | EAGLES  |
| TreeTagger                                             | C++    | Propio  |
| [Stanza](https://stanfordnlp.github.io/stanza/)        | Java   | EAGLES? |
| MaltParser                                             | Java   | EAGLES? |


### Stanza

```python
import stanza

stanza.download('es')

nlp = stanza.Pipeline('es', processors='tokenize,pos,lemma')

doc = nlp("bla bla bla")

df = pd.DataFrame(columns=["text", "lemma", "upos", "xpos", "feats"])
i = 0
for sent in doc.sentences:
    for word in sent.words:
        df.loc[i] = [word.text, word.lemma, word.upos, word.xpos, word.feats]
```



IDEA 2:

Transofrmer Attention solo en el arbol sintatico (parsing)
