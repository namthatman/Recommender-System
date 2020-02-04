Following the Udemy course, "Building Recommender Systems with Machine Learning and AI", by Sundog Education by Frank Kane, Frank Kane

# Recommender System
## 0. Recommender Engine Framework
The architecture of the Recommender Engine building on top of surpriselib.

#### 0.1 EvaluatedAlgorithm(AlgoBase)
EvaluatedAlgorithm contains an algorithm from surpriselib, but introduces a new function called Evaluate that runs all of the metrics in RecommenderMetrics on that algorithm.

    AlgoBase:   SVD, KNNBasic, SVDpp, Custom

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

    Algorithm  RMSE       MAE        HR         cHR        ARHR       Coverage   Diversity  Novelty   
    SVD        0.9034     0.6978     0.0298     0.0298     0.0112     0.9553     0.0445     491.5768  
    Random     1.4385     1.1478     0.0089     0.0089     0.0015     1.0000     0.0719     557.8365  

##
## 1. Content-Based Filtering
The most simple approach, recommending items based on the attributes of those items themselves instead of trying to use aggregate user behavior data. For example, recommend movies in the same genre, has the same actors or directors, etc.

    Content-Based Similarity:   Cosine similarity, Multi-dimensional Cosine, Time Similarity.

#### 1.1 K-Nearest-Neighbors
Measuring the content-based similarity scores between this movie and all others the user rated -> Select/Sort some number, K of the nearest-neighbors to the movie -> Top K nearest movies -> Take the weighted average of similarity scores weighting by the rating the user gave -> Rating prediction.

    Algorithm  RMSE       MAE       
    ContentKNN 0.9375     0.7263    
    Random     1.4385     1.1478 
    
    Using recommender  ContentKNN                                       Using recommender  Random
    We recommend:                                                       We recommend:
    Presidio, The (1988) 3.841314676872932                              Sleepers (1996) 5
    Femme Nikita, La (Nikita) (1990) 3.839613347087336                  Beavis and Butt-Head Do America (1996) 5
    Wyatt Earp (1994) 3.8125061475551796                                Fear and Loathing in Las Vegas (1998) 5
    Shooter, The (1997) 3.8125061475551796                              Happiness (1998) 5
    Bad Girls (1994) 3.8125061475551796                                 Summer of Sam (1999) 5
    The Hateful Eight (2015) 3.812506147555179                          Bowling for Columbine (2002) 5
    True Grit (2010) 3.812506147555179                                  Babe (1995) 5
    Open Range (2003) 3.812506147555179                                 Birdcage, The (1996) 5
    Big Easy, The (1987) 3.7835412549266985                             Carlito's Way (1993) 5
    Point Break (1991) 3.764158410102279                                Wizard of Oz, The (1939) 5

##
## 2. Neighborhood-Based Collaborative Filtering (on-working)
