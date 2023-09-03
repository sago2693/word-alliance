<div align="center">
<h1> Experiments</h1>
</div>

## Introduction
This file provides an overview of our project's experimental design in regards to the performance of our Multitask Bert Model. 

We investigate whether our trained models perform significantly better than the pretrained models and further finetuned for optimal hyperparameters.

## Materials and Methods

### Materials

For training purposes we acquired access to a high performing computing cluster provided by the _Gesellschaft für wissenschaftliche Datenverarbeitung mbH Göttingen_ (GWDG). We were training using a NVIDIA A100-SXM4-80GB with graphical unit (GPU) with CUDA capability.

### Experimental Design

Our experimental design involved creating a randomly initialized model. We employed a pretrained model that we tested against. We trained our data on both the pretrained model and the randomly initialized models to. The primary objectives were to assess the model's effectiveness and optimize hyperparameters for the best results and eventually test the fine-tuned against the pretrained. We seeked to outperform the pretrained model with the fine-tuned one.

### Procedure

We train both pretrained and fine-tuned model simultaneously with different learning rates and batch sizes. For each epoch we save a text file containing meta data to create an overview of the performance. Additionally, only the best model from an epoch is being saved based on the best performance metric.

## Hyperparameter Adjustments
### Batch Size
We made several adjustments to hyperparameters during the project. Notably, we set the learning rate to 1e-05 and experimented with batch sizes. Increasing the batch size beyond 64 led to excessive GPU memory usage and runtime errors.

### Epoch Experimentation
To assess the impact of epochs, we started with a epoch size of 10 and gradually increased it to 50. While larger epoch sizes showed some improvement, the gains were not substantial, especially in tasks related to sentiment analysis. Most notably the best epoch was number 13. 

## Challenges

### Sentiment Analysis
One of the main challenges we encountered was in improving Sentiment Analysis task. Despite our efforts, we found it difficult to significantly enhance performance in this area.

In an attempt to address disparities in dataset sizes, we experimented with annealed sampling. Unfortunately, this approach did not yield promosing results. For instance, for the task of Sentiment Analysis, the results were subpar, with a score of 0.256.

## Conclusion
In this file, we have provided an overview of our project's experimenting, highlighting the adjustments made to hyperparameters, epoch and batch size experimentation, and the challenges faced in improving the Sentiment Analysis tasks. While we have made progress in fine-tuning our model, further refinements are needed to achieve optimal results across all tasks.