<div align="center">
<h1> Results</h1>
</div>

## Introduction 
We are providing an overview of our model's results. We compare a fine-tuned model against a pretrained model while simultaneously testing different hyperparameters. 

## Data
The data were from three different datasets. The Stanford Sentiment Treebank (socher et al., 2013), consisting of 11,855 single sentences from movie reviews extracted from movie reviews, to classify a text's sentiment, the Quora dataset consisting of 400,000 question pairs to train for paraphrase detection and lastly data from the SemEval STS Benchmark dataset consisting of 8,628 sentence pairs of varying similarity for semantic text similarity. 

## Results

Our model achieves the following performance on :

| Model  | Paraphrase Accuarcy  | Sentiment Accuracy | Semantic Text Similarity Correlation |
| ------------------ |---------------- | -------------- | --------------|
| Multitask Classifier BERT Model |     89.01%         |      49.59%       |    88.00%    |
| Pretrained Model |     63.7%         |      26.3%       |    53.4%    | 
| Multitask Classifier Bert Model using Annealed Sampling  |     87.6%         |      42.1%       |    86.8%    | 

During training, we create a text file with information about our best model from every epoch. 

## Conclusion
The results show that our Multitask Classifier BERT Model has significantly outperformed the pretrained model. However, improving the model using Annealed Sampling hasn't yielded favourable results. The loss behaviour was erratic, increasing and decreasing in a large magnitude, probably as a result of using less data from the Quora dataset We were also further limited by the provided hardware to a batch size of 64. 