# PyDRO: 
## A Python reimplementation of the Distributional Random Oversampling method for binary text classification

This repo is a stand-alone (re)implementation of the Distributional Random Oversampling (DRO) method presented in [SIGIR'16](https://dl.acm.org/doi/10.1145/2911451.2914722). 
The former [implementation](https://github.com/AlexMoreo/jatecs/blob/master/src/main/java/it/cnr/jatecs/representation/oversampling/DistributionalRandomOversampling.java) was part of the [JaTeCs](https://github.com/AlexMoreo/jatecs) framework for Java.

Distributional Random Oversampling (DRO) is an oversampling method to counter data imbalance in binary text classification. DRO generates new random minority-class synthetic documents by exploiting the distributional properties of the terms in the collection. The variability introduced by the oversampling method is enclosed in a latent space; the original space is replicated and left untouched.

It comes with a [main](https://github.com/AlexMoreo/pydro/blob/master/src/main.py) file showing an example of how to use it on Reuters-21578.

Reference:
```
@inproceedings{moreo2016distributional,
  title={Distributional Random Oversampling for Imbalanced Text Classification},
  author={Moreo, Alejandro and Esuli, Andrea and Sebastiani, Fabrizio},
  booktitle={SIGIR 2016, 39th ACM Conference on Research and Development in Information Retrieval, Pisa, IT},
  year={2016}
}
```
