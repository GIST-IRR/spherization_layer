# Analogy Test Dataset
This dataset contains five different word analogy datasets. Each contains jsonline files for validation and test, in which each line consists of following dictionary,
```
{"stem": ["raphael", "painter"],
 "answer": 2,
 "choice": [["andersen", "plato"],
            ["reading", "berkshire"],
            ["marx", "philosopher"],
            ["tolstoi", "edison"]]}
```
where `stem` is the query word pair, `choice` has word pair candidates,
and `answer` indicates the index of correct candidate which starts from `0`. Data statistics are summarized as below.

| Dataset | Size (valid/test) | Num of choice | Num of relation group | Original Reference                                                         |
|---------|------------------:|--------------:|----------------------:|:--------------------------------------------------------------------------:|
| sat     | 37/337            | 5             | 2                     | [Turney (2005)](https://arxiv.org/pdf/cs/0508053.pdf)                      |
| u2      | 24/228            | 5,4,3         | 9                     | [EnglishForEveryone](https://englishforeveryone.org/Topics/Analogies.html) |
| u4      | 48/432            | 5,4,3         | 5                     | [EnglishForEveryone](https://englishforeveryone.org/Topics/Analogies.html) |
| google  | 50/500            | 4             | 2                     | [Mikolov et al., (2013)](https://www.aclweb.org/anthology/N13-1090.pdf)    |
| bats    | 199/1799          | 4             | 3                     | [Gladkova et al., (2016)](https://www.aclweb.org/anthology/N18-2017.pdf)   |

All data is lowercased except Google dataset. Please read [our paper](https://arxiv.org/abs/2105.04949) for more information about the dataset and cite it if you use the dataset:
```
@inproceedings{ushio-etal-2021-bert-is,
    title ={{BERT} is to {NLP} what {A}lex{N}et is to {CV}: {C}an {P}re-{T}rained {L}anguage {M}odels {I}dentify {A}nalogies?},
    author={Ushio, Asahi and
            Espinosa-Anke, Luis and
            Schockaert, Steven and
            Camacho-Collados, Jose},
    booktitle={Proceedings of the {ACL}-{IJCNLP} 2021 Main Conference},
    year={2021},
    publisher={Association for Computational Linguistics}
}
```

### LICENSE
The LICENSE of all the resources are under [CC-BY-NC-4.0](./LICENSE). Thus, they are freely available for academic purpose or individual research, but restricted for commercial use.
