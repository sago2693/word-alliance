## IMPORTANT NOTE FOR STS
According to : https://www.sbert.net/examples/training/sts/README.html

To train our network, we need to normalize these scores to a range of 0-1. This can simply be done by dividing the score by 5.
They also suggest a cosine similarity loss.
This article suggests solving STS in a different way:
https://thepythoncode.com/article/finetune-bert-for-semantic-textual-similarity-in-python#models-architecture.
In fact, in the project's documentation the output of this task SEEMS TO BE in the range from 0 to 1
## Multi-task learning setup
Mult-task learning is a hot research topic in which several features of the model and the training process are studied. Here he discuss briefly the following ones: ********************************Architecture, Loss Function and mini-batch configuration********************************

**************************Architecture:**************************

Multiple MTL models for NLP bear a structural similarity to the initial shared architectures found in computer vision. These consist of a communal, a shared global feature extractor succeeded by specialized output branches for various tasks. Unlike Computer Vision, the features correspond to to word representations"   Simply put, it involves a shared word representation, or embeddings, along with a feedforward neural network (FFN) for each task. (Crawshaw, 2020, p. 6) We are following this approach for the architecture to focus more on changes in the training procedure.

**************************Loss function**************************

“The most common approach to train an MTL model is to linearly combine loss functions of different tasks into a single global loss function” (Chen et al., 2021, p. 9) The problem becomes how to choose the weights. The simplest way consists of asigning the same weight for all of them:

```math
L = \sum_{t=1}^{M} \lambda_{t} L_{t} 
```

Given M tasks with their corresponding weights Lambda and Losses L, the simplest way to go about choosing the weights is to set them equal to each other, 𝜆𝑡 = 1/𝑀. To prevent the largest dataset from dominating the others, the weight is set to be inversely proportional to the dataset size.

“since dynamically assigning smaller weights to more uncertain tasks usually leads to good performance for MTL according to [17], [57] assigns weights based on the homoscedasticity of training losses from different tasks” (Chen et al., 2021, p. 9)

$$
\lambda_{t} = \frac{1}{2\sigma_{t}^{2}}
$$

According to (Lauscher et al., 2018) the Loss function with that weighting equals:
$$
The total loss is given by the equation: $$\text{Total loss} = \sum_{t} \frac{1}{2\sigma_{t}^{2}} L_{t} + \ln(\sigma_{t}^{2})$$


Given the project's goal of havingthe best overall performance in the 3 tasks, we will use the dynamic weight assignment to ensure that the gradients are driven by the less noisy task to possibly have a slower but more stable learning.

**************Mini-batches configuration:**************

The typical approach in curriculum learning for transfer learning tasks involves constructing mini-batches that comprise examples specific to an individual task. Subsequently, during training, there is an alternating progression between these tasks. (Worsham and Kalita, 2020, p. 6) 

““(Stickland and Murray, 2019) propose a technique called annealed sampling in which batch sampling is originally based on the ratio of dataset sizes and slowly anneals to an even distribution across all tasks as the current epoch number increases. These discoveries, when combined with curriculum research emerging from the field of reinforcement learning” (Worsham and Kalita, 2020, p. 6) (pdf). The formula of the annealed sampling introduces a factor alpha to which the dataset size must be raised:
$$
\P_i \propto N_i^\alpha
$$

$$
\text{Where}
$$

$$
\alpha = 1 - 0.8 \left( \frac{e - 1}{E - 1} \right)
$$
 Given the considerable difference of sizes between the paraphrase identification dataset and the other two (140.000 vs around 7000), We will use this technique to aim for a balanced distibution of samples at the end of the training.

**************References:**************
Chen, S., Zhang, Y., & Yang, Q. (2021). Multi-Task Learning in Natural Language Processing: An Overview. https://doi.org/10.48550/ARXIV.2109.09138
Crawshaw, M. (2020). Multi-Task Learning with Deep Neural Networks: A Survey. https://doi.org/10.48550/ARXIV.2009.09796
Lauscher, A., Glavaš, G., Ponzetto, S. P., & Eckert, K. (2018). Investigating the Role of Argumentation in the Rhetorical Analysis of Scientific Publications with Neural Multi-Task Learning Models. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 3326–3338. https://doi.org/10.18653/v1/D18-1370
Stickland, A. C., & Murray, I. (2019). BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning. https://doi.org/10.48550/ARXIV.1902.02671
Worsham, J., & Kalita, J. (2020). Multi-task learning for natural language processing in the 2020s: Where are we going? Pattern Recognition Letters, 136, 120–126. https://doi.org/10.1016/j.patrec.2020.05.031
