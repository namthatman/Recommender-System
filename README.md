CREDIT: Udemy course, "Building Recommender Systems with Machine Learning and AI", by Sundog Education by Frank Kane, Frank Kane

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

    Sample result (before tuning):
    Algorithm  RMSE       MAE        HR         cHR        ARHR       Coverage   Diversity  Novelty   
    RBM        1.3257     1.1337     0.0000     0.0000     0.0000     0.0000     0.7423     4551.7024 
    Random     1.4366     1.1468     0.0149     0.0149     0.0041     1.0000     0.0721     552.4610  
    
    Using recommender  RBM                                                          Using recommender  Random
    We recommend:                                                                   We recommend:
    Guardian, The (1990) 2.7926018                                                  Beavis and Butt-Head Do America (1996) 5
    The Boy (2016) 2.788903                                                         Gods Must Be Crazy, The (1980) 5
    Super, The (1991) 2.7884843                                                     Seven (a.k.a. Se7en) (1995) 5
    Christmas Vacation (National Lampoon's Christmas Vacation) (1989) 2.7860935     Reality Bites (1994) 5
    Bright Eyes (1934) 2.784541                                                     Young Guns (1988) 5
    Vicky Cristina Barcelona (2008) 2.7843983                                       Fear and Loathing in Las Vegas (1998) 5
    That's My Boy (2012) 2.7839901                                                  Pet Sematary (1989) 5
    Troy (2004) 2.783449                                                            Ghostbusters (a.k.a. Ghost Busters) (1984) 5
    Dear White People (2014) 2.7833507                                              Requiem for a Dream (2000) 5
    Stargate: The Ark of Truth (2008) 2.7827556                                     Herbie Rides Again (1974) 5

#### 4.2 Auto-Encoders for Recommendations (Autorec)
Autorec has three layers: an input layer that contains individual ratings, a hidden layer, and an output layer that gives us our predictions. A matrix of weights between the layers is maintained across every instance of this network, as well as a bias node for both hidden and output layers.

    Sample result (before tuning):
    Algorithm  RMSE       MAE        HR         cHR        ARHR       Coverage   Diversity  Novelty   
    AutoRec    1.8253     1.4222     0.0075     0.0075     0.0026     1.0000     0.0713     512.4595 
    Random     1.4366     1.1468     0.0149     0.0149     0.0041     1.0000     0.0721     552.4610  
    
    Using recommender  RBM                                                          Using recommender  Random
    We recommend:                                                                   We recommend:
    Deer Hunter, The (1978) 5                                                       Beavis and Butt-Head Do America (1996) 5
    Seven (a.k.a. Se7en) (1995) 5                                                   Gods Must Be Crazy, The (1980) 5
    Circle of Friends (1995) 5                                                      Seven (a.k.a. Se7en) (1995) 5
    Raiders of the Lost Ark (Indiana Jones & the Raiders of the Lost Ark) (1981) 5  Reality Bites (1994) 5
    Rocketeer, The (1991) 5                                                         Young Guns (1988) 5
    Arachnophobia (1990) 5                                                          Fear and Loathing in Las Vegas (1998) 5
    Hero (Ying xiong) (2002) 5                                                      Pet Sematary (1989) 5
    Rob Roy (1995) 5                                                                Ghostbusters (a.k.a. Ghost Busters) (1984) 5
    Thin Red Line, The (1998) 5                                                     Requiem for a Dream (2000) 5
    October Sky (1999) 5                                                            Herbie Rides Again (1974) 5

#### 4.3 Session-Based Recommendations with RNNs (GRU4Rec)
The architecture of GRU4Rec: 
    
    input layer (one-hot encoded item) -> embedding layer -> gru layers -> feedforward layers -> output scores on items
    
represents the processing of a single event in the clickstream. We start with the item that was just viewed encoded as 1-of-N, and that goes into an embedding layer, and that in turn leads into multiple GRU layers then multiple feedforward layers. And finally we get scores on the all of the items, from which we can select the items the deep network thinks is most likely to be viewed next in the clickstream.

    Result from training on MovieLens
    Recall@20: 0.11595807746943615  MRR@20: 0.029791653819438553

Unfortunately, the ouput isn't terribly interesting because it would be hard to subjectively evaluate the results given that we are using a dataset that wasn't really intended for this to begin with. More information on ICLR'2016 paper "Session-based Recommendations With Recurrent Neural Networks". See paper: http://arxiv.org/abs/1511.06939.

##
## 5. Real-World Challenges of Recommender Systems
#### 5.1 The Cold Start Problem
The Cold Start Problem is as when a brand-new user arrives at your site, what do you recommend to them when you know nothing about them yet? The same for new items in your catalog, how do they get recommended when there is no data on them yet to pair with other items?

Cold-Start: new user solutions

    +use implicit data: a new user's interests can be indicated as soon as they looks at an item on your site, you'll have at least some implicit information about this user's interests.
    +use cookies (carefully): a new user or an existing user who hasn't logged in yet, use browser cookies to help identify users even when they are logged out, and tie the user session to a user account for use making recommendation.
    +geo-ip: identify user with IP address they are connecting from, many IP addresses can be translated into geographical locations. Then recommend based on other people from the same region or popular items from this region.
    +recommend top-sellers or promotions: a safer bet is recommend top-selling items or promotions when you have nothing else to work with.
    +interview the user: ask user's interests to get some information to work with on personal interests.
    
Cold-Start: new item solutions
    
    +just don't worry about it: it can be discovered in search results, or appear in promotions
    +use content-based attributes: recommend it alongside items that have similar attributes.
    +map attributes to latent features: merge content attributes with latent features learned rating behavior pattern through matrix factorization, or deep learning. (see paper called LearnAROMA)
    +random exploration: dedicate extra slots in top-N recommendations to randomly showing new items to users, in an attempt to gather more data on them.
    
#### 5.2 Stoplists
Stoplists is checking to see if an item might cause unwanted controversy before you let it into your recommender system.

Things you might stoplist

    adult-oriented content
    vulgarity
    legally prohibited topics (i.e. Mein Kampf)
    terrorism / political extremism
    bereavement / medical
    competing products
    drug use
    religion
    
Stoplists should be updated and applied quickly should the need arise.

#### 5.3 Filter Bubbles Trust
Filter Bubbles refers to societal problems that arise when all you show people are things that appeal to their existing interests. This is called a filter bubble because the content you're presenting your users is filtered in such a way that it keeps them within a bubble of their pre-existing interests. Transparency, allow the user to see exactly why you recommended this item to them, and ideally, let them fix the root cause themselves. It's a much better outcome if a user understands why you recommended something that they found confusing.

#### 5.4 Outliers
Outliers, some users in your data aren't actually real people, but are bots that are rating things in an unnatural manner. A bot might also generate an exessively large number of ratings, and end up having a very large influence on you collaborative filtering recommendations. Even among real peopel, you might want to filter some of them out, i.e. professional reviewers, institutional buyers, who can have a huge influence on your recommendations.

#### 5.5 Gaming The System
Another real-world problem is people trying to game your system. If items your recommender system promotes leads to those items being purchased more, the makers of those items have a financial incentive to find ways to game your system into recommending their items more often. Or people with certain ideological agendas might purposelt try to make your ssytem recommend items that promote their own ideology, or to not recommend items that run counter to it. Some hacker might even be bored, and try to create humorous pairings in your recommender system just for their own amusement.

One techique that works remarkably well for this problem, make sure that recommendations are only generated from people who actually spent real money on the item. Or make sure that you only allow reviews from people you know actually purchased or cunsumed the content in question.

#### 5.6 Fraud The Perils of Clickstream
Using implicit clickstream data, such as images people click on, is fraught with problems. You should always be extremly skeptical about building a recommender system that relies only on things people click on, such as ads. Not only are these sorts of systems highly susceptible to gaming, they are susceptible to quirks of human behavior that aren't useful for recommendations. Clickstream data is a very unreliable signal of interest. What people click on and what they buy can be very diffent things.

#### 5.7 International Markets and Laws
If your recommender system spans customers in different countries, there may be specific challenges that you need to consider. International customer data when training a recommender system, in most cases, should be kept separated by country. Since you don't want to recommend items in a foreign language to people who don't speak that language, and there may be cultural differences that influence peoples' tastes in different countries.

There is also the problem of availability and content restrictions. Some countries have legal restrictions on what sort of content can be consumed as well, which must be taken into consideration. Since your recommender system depends on collectiong data on individual interest, there are also privacy laws to take into consideration, and these too vary by country.

#### 5.8 Temporal Effects
Dealing with the effects of time, one example is seasonality, some items, like Christmas decorations, only make for good recommendations just before Christmas. Or recommending bikinis in the winter is also a bad idea. Picking up on annual patterns like this is hard to do, and most recommender systems won't do it automatically. But, something you can do more easily and more generally is taking the recency of a rating into account. As peoples' tastes change quickly, and a rating a user made yesterday is much stronger indication of their interest than a rating they made a year ago. The product a user is looking at right now is the most powerful indicator of their current interest. It makes sense to use rating recency as a training feature of its own.

#### 5.9 Value-Aware Recommendations
In the real world, a recommender system you're building for a company will ultimately exist for the purpose of driving their profit. In the real world, some items are more profitable than others. Some items might even be offered at below cost as loss leaders to attract people to your site, and recommending them actually costs your company money. This presents a bit of a moral quandary for the recommender system developers. Do we keep our algorithms pure and optimize only for users' interests, or do we work profit into the equation by rewarding algorithms that generate more profit? This is value-aware recommendations, where the contribution to the company's bottom line plays a role in whether something is recommended.

Optimizing too much for profit can backfire. You don't want to end up only recommending expensive items to yor customers, because they will be less likely to purchase them. It might make more sense to look at profit margin, and not actual profit, so the cost of the item isn't really a factor in what you recommend.
