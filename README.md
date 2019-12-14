# Classical Chinese Quatrain Generator based on the RNN encoder-decoder model

A planning-based architecture is implemented based on [Wang et al. 2016](https://arxiv.org/abs/1610.09889).
Some basic code structures are based on https://github.com/DevinZ1993/Chinese-Poetry-Generation.

## Code Organization

![Structure of Code](img/structure.jpg)



## Dependencies

* Python 3.7

* [Numpy](http://www.numpy.org/)

* [TensorFlow2](https://www.tensorflow.org/)

* [Jieba](https://github.com/fxsjy/jieba)

* [Gensim](https://radimrehurek.com/gensim/)


## Data Processing

Run the following command to generate training data from source text data:

    ./data_utils.py

Depending on your hardware, this can take you a cup of tea or over one hour.
The keyword extraction is based on the TextRank algorithm,
which can take a long time to converge.

## Training

The poem planner was based on Gensim's Word2Vec module.
To train it, simply run:

    ./train.py -p

The poem generator was implemented as an enc-dec model with attention mechanism.
To train it, type the following command:

    ./train.py -g

You can also choose to train the both models altogether by running:

    ./train.py -a

To erase all trained models, run:

    ./train.py --clean


The average loss will converge at ~3.8.

## Generating

Type the following command:

    ./main.py

Please type in a hint text in Chinese, it should return a quatrain poem.


## Improve It

* Apply a new method to better tuning rhythm and tone. Our current implementation is to penalize the probability of selecting character that is out of the rhythm in the generating process.

* To improve the planning model,
Please refine the planner class in [plan.py](./plan.py).

* To improve  the generation model,
Please refine the generator class in [generate.py](./generate.py).

