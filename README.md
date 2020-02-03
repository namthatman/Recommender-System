Following the Udemy course, "Building Recommender Systems with Machine Learning and AI", by Sundog Education by Frank Kane, Frank Kane

# Recommender System
## 0. Recommender Engine Framework
The architecture of the Recommender Engine building on top of surpriselib.

#### 0.1 EvaluatedAlgorithm(AlgoBase)
EvaluatedAlgorithm contains an algorithm from surpriselib, but introduces a new function called Evaluate that runs all of the metrics in RecommenderMetrics on that algorithm.

#### 0.2 RecommenderMetrics
RecommenderMetrics contains evaluating functions to measure:

    RMSE:      Root Mean Squared Error. Lower values mean better accuracy.
    MAE:       Mean Absolute Error. Lower values mean better accuracy.
    HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.
    cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.
    ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better.
    Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.
    Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations
               for a given user. Higher means more diverse.
    Novelty:   Average popularity rank of recommended items. Higher means more novel.
    
#### 0.3 EvaluationData(Dataset)
EvaluationData takes in a Dataset, which might come from MovieLens that loads data, and creates all the train/test splits needed by EvaluatedAlgorithm.

#### 0.4 MovieLens
MovieLens loads MovieLens dataset and performs preprocessing and cleaning.

#### 0.5 Evaluator
Evaluator compares the performance of different recommender algorithms against each other by adding algorithms that needed to evaluate into the class.

#### 0.6 RecsBakeOff
RecsBakeOff is the main class to run the other class of the Recommender Engine.


## 1. Content-Based Filtering
on-working
