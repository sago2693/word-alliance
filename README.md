# DNLP SS23 Final Project - Multitask BERT

This is the starting code for the default final project for the Deep Learning for Natural Language Processing course at the University of Göttingen. You can find the handout [here](https://1drv.ms/b/s!AkgwFZyClZ_qk718ObYhi8tF4cjSSQ?e=3gECnf)

In this project, you will implement some important components of the BERT model to better understanding its architecture. 
You will then use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection and semantic similarity.

After finishing the BERT implementation, you will have a simple model that simultaneously performs the three tasks.
You will then implement extensions to improve on top of this baseline.

## Setup instructions

* Follow `setup.sh` to properly setup a conda environment and install dependencies.
* There is a detailed description of the code structure in [STRUCTURE.md](./STRUCTURE.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, external libraries that give you other pre-trained models or embeddings are not allowed (e.g., `transformers`).

## Handout

Please refer to the handout for a through description of the project and its parts.

### Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!) 

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

## Multi-task learning setup
Mult-task learning is a hot research topic in which several features of the model and the training process are studied. Here he discuss briefly the following ones: ********************************Architecture, Loss Function and mini-batch configuration********************************

**************************Architecture:**************************

Multiple MTL models for NLP bear a structural similarity to the initial shared architectures found in computer vision. These consist of a communal, a shared global feature extractor succeeded by specialized output branches for various tasks. Unlike Computer Vision, the features correspond to to word representations"   Simply put, it involves a shared word representation, or embeddings, along with a feedforward neural network (FFN) for each task. (Crawshaw, 2020, p. 6) We are following this approach for the architecture to focus more on changes in the training procedure.

**************************Loss function**************************

“The most common approach to train an MTL model is to linearly combine loss functions of different tasks into a single global loss function” (Chen et al., 2021, p. 9) The problem becomes how to choose the weights. The simplest way consists of asigning the same weight for all of them:

$$
L = \sum_{t=1}^{M} \lambda_{t} L_{t} 

$$

Given M tasks with their corresponding weights Lambda and Losses L, the simplest way to go about choosing the weights is to set them equal to each other, 𝜆𝑡 = 1/𝑀. To prevent the largest dataset from dominating the others, the weight is set to be inversely proportional to the dataset size.

“since dynamically assigning smaller weights to more uncertain tasks usually leads to good performance for MTL according to [17], [57] assigns weights based on the homoscedasticity of training losses from different tasks” (Chen et al., 2021, p. 9)

$$
\lambda_{t} = \frac{1}{2\sigma_{t}^{2}}

$$
Given the project's goal of havingthe best overall performance in the 3 tasks, we will use the dynamic weight assignment to ensure that the gradients are driven by the less noisy task to possibly have a slower but more stable learning.

**************Mini-batches configuration:**************

The typical approach in curriculum learning for transfer learning tasks involves constructing mini-batches that comprise examples specific to an individual task. Subsequently, during training, there is an alternating progression between these tasks. (Worsham and Kalita, 2020, p. 6) 

““(Stickland and Murray, 2019) propose a technique called annealed sampling in which batch sampling is originally based on the ratio of dataset sizes and slowly anneals to an even distribution across all tasks as the current epoch number increases. These discoveries, when combined with curriculum research emerging from the field of reinforcement learning” (Worsham and Kalita, 2020, p. 6) (pdf) Giventhe considerable difference of sizes between the paraphrase identification dataset and the other two (140.000 vs around 7000), We will use this technique to aim for a balanced distibution of samples at the end of the training.

**************References:**************
Chen, S., Zhang, Y., & Yang, Q. (2021). Multi-Task Learning in Natural Language Processing: An Overview. https://doi.org/10.48550/ARXIV.2109.09138
Crawshaw, M. (2020). Multi-Task Learning with Deep Neural Networks: A Survey. https://doi.org/10.48550/ARXIV.2009.09796
Stickland, A. C., & Murray, I. (2019). BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning. https://doi.org/10.48550/ARXIV.1902.02671
Worsham, J., & Kalita, J. (2020). Multi-task learning for natural language processing in the 2020s: Where are we going? Pattern Recognition Letters, 136, 120–126. https://doi.org/10.1016/j.patrec.2020.05.031
