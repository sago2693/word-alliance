
## Multi-task learning setup
Mult-task learning is a training paradigm that aim at exploiting similarities between tasks to enhance their individual performance (Crawshaw, 2020). Implementing systems of this kind requires to consider aspects like the architecture design, the loss function and the batch configuration.


**************************Architecture:**************************

Multiple MTL models for NLP bear a structural similarity to the initial shared architectures found in computer vision. These consist of a communal, a shared global feature extractor succeeded by specialized output branches for various tasks. Unlike Computer Vision, the features correspond to to word representations. Simply put, it involves a shared word representation, or embeddings, along with a feedforward neural network (FFN) for each task. (Crawshaw, 2020) We are following this approach for the architecture to focus more on changes in the training procedure of the current datasets rather than training each task separately

**************************Loss function**************************

“The most common approach to train an MTL model is to linearly combine loss functions of different tasks into a single global loss function” (Chen et al., 2021, p. 9) The challenge now lies in determining the appropriate weights. A straightforward approach involves assigning equal weight to each of them:

```math
L = \sum_{t=1}^{M} \lambda_{t} L_{t} 
```
Given M tasks with their corresponding weights Lambda and Losses L, the simplest way to go about choosing the weights is to set them equal to each other, 𝜆𝑡 = 1/𝑀. To prevent the largest dataset from dominating the others, the weight is set to be inversely proportional to the dataset size.

Chen et al (2021) found that "dynamically assigning smaller weights to more uncertain tasks usually leads to good performance for MTL”. Based on this hypothesis, we create a dynamic weight lambda using the following formula:

$$
\lambda_{t} = \frac{1}{2\sigma_{t}^{2}}
$$

According to Lauscher et al. (2018) the Loss function with that weighting equals:
 $$\text{Total loss} = \sum_{t} \frac{1}{2\sigma_{t}^{2}} L_{t}$$


We implemented this dynamic weighting expecting to have losses with stable behaviours that lead to a better overall performance.

**************Mini-batches configuration:**************

The typical approach in curriculum learning for transfer learning tasks involves constructing mini-batches that comprise examples specific to an individual task. Subsequently, during training, there is an alternating progression between these tasks. (Worsham and Kalita, 2020) 

Stickland and Murray(2019) propose a technique called annealed sampling in which batch sampling is originally based on the ratio of dataset sizes and slowly anneals to an even distribution across all tasks as the current epoch number increases". The formula of the annealed sampling introduces a factor alpha to which the dataset size must be raised to a factor alpha:

The proportion P_i is given by: 

$$
P_i ∝ N_i^\alpha
$$

Where α is defined as: 

$$
\alpha = 1 - 0.8 \left( \frac{e - 1}{E - 1} \right)
$$

 Given the considerable difference of sizes between the paraphrase identification dataset and the other two (140.000 vs around 7000), We use this technique to aim for a balanced distibution of samples at the end of the training.

## Learning rate scheduling 
Another approach to improve the results was implementing learning rate scheduling. We used StepLR which reduces learning rate every epoch (set by step size after how many epochs we reduce the learning rate). It is done in order to prevent overfitting and overcome possible overshooting an optimum. It allows the model to train more efficiently. Since we already start with small learning rate we used 0.9 gamma factor to not reduce the learning rate too much. we tried different values, but in the end it didn't improve results significantly. 

## Note for STS
Based on [a Sentence transformer implementation](https://www.sbert.net/examples/training/sts/README.html) we train our network normalizing the textual similarity scores to a range of 0-1. This can simply be done by dividing the score by 5.

**************References:**************
- Chen, S., Zhang, Y., & Yang, Q. (2021). Multi-Task Learning in Natural Language Processing: An Overview. https://doi.org/10.48550/ARXIV.2109.09138
- Crawshaw, M. (2020). Multi-Task Learning with Deep Neural Networks: A Survey. https://doi.org/10.48550/ARXIV.2009.09796
- Lauscher, A., Glavaš, G., Ponzetto, S. P., & Eckert, K. (2018). Investigating the Role of Argumentation in the Rhetorical Analysis of Scientific Publications with Neural Multi-Task Learning Models. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 3326–3338. https://doi.org/10.18653/v1/D18-1370
- Stickland, A. C., & Murray, I. (2019). BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning. https://doi.org/10.48550/ARXIV.1902.02671
- Worsham, J., & Kalita, J. (2020). Multi-task learning for natural language processing in the 2020s: Where are we going? Pattern Recognition Letters, 136, 120–126. https://doi.org/10.1016/j.patrec.2020.05.031
