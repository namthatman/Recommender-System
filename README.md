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

    Sample result:
    Algorithm  RMSE       MAE        HR         cHR        ARHR       Coverage   Diversity  Novelty   
    SVD        0.9034     0.6978     0.0298     0.0298     0.0112     0.9553     0.0445     491.5768  
    Random     1.4385     1.1478     0.0089     0.0089     0.0015     1.0000     0.0719     557.8365  

##
## 1. Content-Based Filtering
The most simple approach, recommending items based on the attributes of those items themselves instead of trying to use aggregate user behavior data. For example, recommend movies in the same genre, has the same actors or directors, etc.

    Content-Based Similarity:   Cosine similarity, Multi-dimensional Cosine, Time Similarity.

#### 1.1 K-Nearest-Neighbors
Measuring the content-based similarity scores between this movie and all others the user rated -> Select/Sort some number, K of the nearest-neighbors to the movie -> Top K nearest movies -> Take the weighted average of similarity scores weighting by the rating the user gave -> Rating prediction.

    Sample result:
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
## 2. Neighborhood-Based Collaborative Filtering
This is the idea of leveraging the behavior of the others to inform what you might enjoy. At the very high level, it means finding other people like you and recommending stuff they liked. Or it might mean finding other things similar to the thing that you like. That is, recommending stuff people bought who also bought the stuff that you liked. Recommending stuff based on other people's collaborative behavior.

    Neighborhood-Based Similarity:   Cosine, Adjusted Cosine, Pearson Similarity, Mean Squared Differencce, Jaccard Similarity, Spearman Rank Correlation.
    
#### 2.1 User-Based Collaborative Filtering
Start by finding other users similar to yourself, based on their ratings history, and recommend stuff they liked that you haven't seen yet. Steps: user-item rating matrix -> user-user similarity matrix -> look up top similar users -> candidate generation -> candidate scoring -> candidate filtering.

    Sample result:
    testSubject = '85'
    We recommend:
    Inception (2010) 3.3
    Star Wars: Episode V - The Empire Strikes Back (1980) 2.4
    Bourne Identity, The (1988) 2.0
    Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 2.0
    Dark Knight, The (2008) 2.0
    Good, the Bad and the Ugly, The (Buono, il brutto, il cattivo, Il) (1966) 1.9
    Departed, The (2006) 1.9
    Dark Knight Rises, The (2012) 1.9
    Back to the Future (1985) 1.9
    Gravity (2013) 1.8
    Fight Club (1999) 1.8

#### 2.2 Item-Based Collaborative Filtering
Instead of looking for other people similar to you, look at the things you liked and recommend stuff that's similar to those things. Using similarities between items could be better than similarities between people, items tend to be of a more permanent nature than people whose tastes may change quickly.  

    Sample result:
    testSubject = '85'
    We recommend:
    James Dean Story, The (1957) 10.0
    Get Real (1998) 9.987241120712646
    Kiss of Death (1995) 9.966881877751941
    Set It Off (1996) 9.963732215657119
    How Green Was My Valley (1941) 9.943984081065269
    Amos & Andrew (1993) 9.93973694500253
    My Crazy Life (Mi vida loca) (1993) 9.938290487546041
    Grace of My Heart (1996) 9.926255896645218
    Fanny and Alexander (Fanny och Alexander) (1982) 9.925699671455906
    Wild Reeds (Les roseaux sauvages) (1994) 9.916226404418774
    Edge of Seventeen (1998) 9.913028764691676

#### 2.3 User-Based KNN Recommender
Find the K most-similar users who rated this item -> compute mean similarity score weighted by ratings -> rating prediction.

#### 2.4 Item-Based KNN Recommender
Find the K most-similar items also rated by this user -> compute mean similarity score weighted by ratings -> rating prediction.

    Sample result:
    Algorithm  RMSE       MAE       
    User KNN   0.9961     0.7711    
    Item KNN   0.9995     0.7798    
    Random     1.4385     1.1478   

    Using recommender  User KNN                                 Using recommender  Item KNN                                 Using recommender  Random                                     
    We recommend:                                               We recommend:                                               We recommend:
    One Magic Christmas (1985) 5                                Life in a Day (2011) 5                                      Sleepers (1996) 5
    Step Into Liquid (2002) 5                                   Under Suspicion (2000) 5                                    Beavis and Butt-Head Do America (1996) 5
    Art of War, The (2000) 5                                    Asterix and the Gauls (AstÃ©rix le Gaulois) (1967) 5        Fear and Loathing in Las Vegas (1998) 5
    Taste of Cherry (Ta'm e guilass) (1997) 5                   Find Me Guilty (2006) 5                                     Happiness (1998) 5
    King Is Alive, The (2000) 5                                 Elementary Particles, The (Elementarteilchen) (2006) 5      Summer of Sam (1999) 5
    Innocence (2000) 5                                          Asterix and the Vikings (AstÃ©rix et les Vikings) (2006) 5  Bowling for Columbine (2002) 5
    MaelstrÃ¶m (2000) 5                                         From the Sky Down (2011) 5                                  Babe (1995) 5
    Faust (1926) 5                                              Vive L'Amour (Ai qing wan sui) (1994) 5                     Birdcage, The (1996) 5
    Seconds (1966) 5                                            Vagabond (Sans toit ni loi) (1985) 5                        Carlito's Way (1993) 5
    Amazing Grace (2006) 5                                      Ariel (1988) 5                                              Wizard of Oz, The (1939) 5

##
## 3. Matrix Factorization Methods
Model-Based methods, since instead of trying to find items or users that are similar to each other, we'll instead apply data science and machine learning techniques to extract predictions from our ratings data. We will train models with user rating data and use those models to predict the ratings of new items by users.

Matrix Factorization algorithms manage to find broader features of users and items on their own, like action movies or romantic. Although the math doesn't know what to call them, they are just described by matrices that describe whatever attributes fall out of the data. The general idea is to describe users and movies as combinations of different amounts of each feature.

#### 3.1 Principal Component Analysis (PCA)
PCA, as a dimensionality reduction problem, that is, we want to take data that exists in many dimensions, like all of the movies a user might rate, into a smaller set of dimensions that can accurately describe a movie, such as its genres.

#### 3.2 Singular Value Decomposition (SVD)
SVD is a way to computing Matrix Factorization at once very efficiently. SVD runs PCA on both the users and the items, and gives back the matrices we need that are factors of the ratings matrix we want.

    Sample result:
    Algorithm  RMSE       MAE       
    SVD        0.9039     0.6984    
    SVD++      0.8943     0.6887    
    Random     1.4359     1.1493    
    
    Using recommender  SVD                                      Using recommender  SVD++                                    Using recommender  Random                                     
    We recommend:                                               We recommend:                                               We recommend:
    Gladiator (1992) 4.520884890007874                          Lock, Stock & Two Smoking Barrels (1998) 4.6042276662762    Usual Suspects, The (1995) 5
    Philadelphia Story, The (1940) 4.420701711947352            The Imitation Game (2014) 4.457817607681913                 Legends of the Fall (1994) 5
    Stand by Me (1986) 4.3959589752178365                       Amadeus (1984) 4.322529285260794                            Trainspotting (1996) 5
    Moon (2009) 4.372613693384055                               Indiana Jones and the Last Crusade (1989) 4.289355864317    Titanic (1997) 5
    Happiness (1998) 4.369493252705134                          Happiness (1998) 4.260333724293291                          Happiness (1998) 5
    American Graffiti (1973) 4.353470600109924                  Harry Potter and the Sorcerer's Stone (2001) 4.229511982    Big Daddy (1999) 5
    And Your Mother Too (Y tu mamÃ¡ tambiÃ©n) (2001) 4.34915    Wings of Desire (Himmel Ã¼ber Berlin, Der) (1987) 4.2242    Spider-Man 2 (2004) 5
    Wallace & Gromit: A Close Shave (1995) 4.315441215430408    Grand Day Out with Wallace and Gromit, A (1989) 4.223352    Blade Runner (1982) 5
    Band of Brothers (2001) 4.315414828016616                   Charade (1963) 4.208953110263677                            Stand by Me (1986) 5
    Seven Samurai (Shichinin no samurai) (1954) 4.3111029206    Snatch (2000) 4.204477673848874                             Indiana Jones and the Last Crusade (1989) 5

##
## 4. Deep Learning for Recommender Systems
Deep Learning can be very good at recognizing patterns at a way similar to how human brain may do it. It's good at things like image recognition and predicting sequences of events. Thus, you can think of recommender systems as looking for very complex patterns based on the behavior of the other people. So, Matrix Factorization can be modeled as a neural network.

#### 4.1 Restricted Boltzmann Machines (RBMs)
The general idea is to use each individual user in our training data as a set of inputs into RBM to train it. Process each user as part of a batch during training, looking at their ratings for every movie they rated. Visible nodes represent ratings for a given user on every movie, and we are trying to learn weights and biases to reconstruct ratings for user-movie pairs we don't know yet. As we are training our RBM with a given user's known ratings, we attempt to learn the weights and biases used for the movies that user actually rated. As we iterate through training on all of the other users, we fill in the other weights and biases as we go.

#### 4.2 Auto-Encoders for Recommendations (Autorec)
Autorec has three layers: an input layer that contains individual ratings, a hidden layer, and an output layer that gives us our predictions. A matrix of weights between the layers is maintained across every instance of this network, as well as a bias node for both hidden and output layers.







