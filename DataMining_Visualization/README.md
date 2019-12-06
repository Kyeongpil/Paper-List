# DataMining & Visualization

### DataMining

-   [Deep content-based music recommendation](https://papers.nips.cc/paper/5004-deep-content-based-music-recommendation)

    -   CNN 을 통해 음악 feature 를 extract
    -   Weighted Matrix Factorization 을 통해 추천

*   [Collaborative Feature Learning from Social Media](https://arxiv.org/abs/1502.01423)

    -   이미지는 CNN 을 통해 feature 를 뽑음
    -   Matrix factorization(with Negative sampling)을 통해 Collaborative filtering 수행

-   [Learning Topics in Short Texts by Non-negative Matrix Factorization on Term Correlation Matrix](http://epubs.siam.org/doi/abs/10.1137/1.9781611972832.83)

    -   term occurrence 를 PPMI(Positive Pointwise Mutual Information) matrix 만들고
    -   NMF(Non-negative matrix factorization)을 통해 토픽 모델링 수행

*   [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)

    -   Neural Network 에 deep 하게 쌓은 network 와 wide 한 network 결합
    -   딥러닝을 추천 시스템에 사용
    -   Training 크기 500 Billion ...

-   [Learning Entity and Relation Embeddings for Knowledge Graph Completion](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/aaai2015_transr.pdf)

    -   Knowlege graph embedding 모델로 TransR 제안
    -   entity space 와 relation space 를 같은 공간에 embedding 함
    -   relation 을 클러스터링하여 관계를 그룹핑한 CTransR 버전

*   [Knowledge Graph Embedding for Hyper-Relational Data](http://ieeexplore.ieee.org/document/7889640/)

    -   Hyper-relation(두 entity 간 다중의 relation 이 있을 수 있음) 문제를 해결하기 위한 모델 제안
    -   transHR, hyper-relation 에 대해 각각의 relation matrix 를 통해 embedding 함

-   [TransG: A Generative Model for Knowledge Graph Embedding](https://aclweb.org/anthology/P/P16/P16-1219.pdf)

    -   multiple relation 문제 -> relation 에 클러스터링 도입
    -   클러스터링 갯수를 모름 -> Chinese restaurant process 로 동적 생성

*   [Learning to Extract Conditional Knowledge for Question Answering Using Dialogue](http://dl.acm.org/citation.cfm?id=2983777)

    -   대화를 할 때 conditional knowledge base(CKB)를 추출
    -   clustering 을 통해 answer 를 할 때 해당 CKB 와 관련이 있는 답을 하도록 학습

-   [Augmented Variational Autoencoders for Collaborative Filtering with Auxiliary Information](https://dl.acm.org/citation.cfm?id=3132972)

    -   추천시스템에 Variational autoencoder 와 Adversarial regularization 을 융합한 모델 제안
    -   이외에 기존의 여러 종류의 VAE 모델도 적용

*   [Latent Relational Metric Learning via Memory-based Attention for Collaborative Ranking](https://www.researchgate.net/publication/322385353_Latent_Relational_Metric_Learning_via_Memory-based_Attention_for_Collaborative_Ranking)

    -   추천시스템에 Knowledge graph embedding 기법과 딥러닝의 attention(memory) mechanism 도입
    -   각 user 와 item 간의 relation vector 도입

*   [Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding](https://dl.acm.org/citation.cfm?id=3159656)

    -   순차적으로 본 아이템 정보를 통해 유저가 그 다음 어떤 아이템을 볼지를 예측하는 태스크
    -   CNN 을 통해 Vertical filter 와 Horizontal filter 들을 사용
    -   user embedding 도 추가로 사용

* [Outer Product-based Neural Collaborative Filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/ijcai18-ConvNCF.pdf)

    - user embedding vector와 item embedding vector를 outer product하여 interaction map 생성
    - interaction map에 대해서 CNN layer를 통과하여 해당 유저가 해당 아이템을 볼 확률 생성
    - Bayesian Personalized Ranking loss 사용

* [Exploiting Tri-Relationship for Fake News Detection](https://arxiv.org/abs/1712.07709)

    - 언론사와 신문, 신문과 유저, 유저와 유저 간의 관계를 고려
    - 각 관계를 NMF (Nonnegative matrix factorization) 방식을 이용하여 embedding 후 신문의 임베딩 벡터를 linear classifier로 semi-supervised 방식으로 학습함

* [Ranking Distillation: Learning Compact Ranking Models With High Performance for Recommender System](http://www.sfu.ca/~jiaxit/resources/kdd18ranking.pdf)

    - 먼저 큰 규모의 모델(Teacher model)을 학습한 뒤, 데이터셋과 큰 모델이 예측한 확률분포를 통해 작은 모델(Student model)을 학습
    - 이를 통해 Student model은 Teacher model에서 나온 Ranking (Top-K) 및 데이터셋의 latent 분포를 배울 수 있도록 학습함 
    - Overfitting에 강하고 좀더 효율적으로 추론이 가능

* [RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems](https://arxiv.org/abs/1803.03467)

    - Click set (Implicit set)과 knowledge graph data를 결합한 추천시스템 모델
    - graph 상에서 링크에 따라 propagate를 하여 neighbor들을 확장한 rippleset을 이용하여 user embedding vector를 계산함, 이 때 knowledge graph embedding 기법과 attention 기법을 활용하여 각 노드들을 임베딩함
    - knowledge graph 정보까지 활용하여 영화 간의 정보 (국가, 감독 등)을 효과적으로 사용

* [Adversarial Personalized Ranking for Recommendation](https://arxiv.org/abs/1808.03908)

    - Collaborative filtering에 adversarial learning (adversarial noise)를 적용
    - Bayesian personalized ranking (BPR) loss 적용
    - adversarial noise를 활용한 regularization을 통해 l2 regularization 등의 기법을 적용하여 모델 파라미터 가중치 크기를 줄이지 않더라도 overfitting을 방지함 (generalization에 robust함)


### Social mining

-   [The spread true and false news online](http://science.sciencemag.org/content/359/6380/1146)

    -   Fake news detection 관련
    -   여러 fake news detection site 에서 fake, true 뉴스 링크를 통해 트위터에서 해당 링크를 공유하는 트윗을 수집
    -   해당 트윗들을 리트윗(spread, cascade)하는 범위(depth, breadth) 및 속도, 갯수를 조사
    -   fake news 가 true news 보다 더 깊고 넓게 확산됨
    -   그 이유를 몇가지 연구를 통해 분석
    -   막상 fake news 를 퍼뜨리는 user 는 follower 수가 그리 많지는 않음
    -   fake news 들은 대체적으로 novelty attract 가 높음
    -   감정분석을 한 결과 fake news 와 true news 가 갖는 감정이 대체적으로 다름

*   [What is Twitter, a Social Network or a News Media?](https://dl.acm.org/citation.cfm?id=1772751)

    -   트위터에서의 social network 분석
    -   follow, retweet 관계 등에 따른 user 영향도 등 분석
    -   토픽의 트렌드가 어떻게 퍼지고 지속되는지도 분석함

* [Automatic Opinion Leader Recognition in Group Discussions](https://ieeexplore.ieee.org/document/7880177/)
    
    - 그룹 내에서의 speech signal을 통해 오피니언 리더가 누군지 판별하는 알고리즘
    - 얼마나 감정적인 대화를 하는지에 대한 Emotion ratio와 얼마나 대화에서 비중있게 말하는지에 대한 Conversation ratio를 통해 opnion leader의 score를 계산

* [Detecting Opinion Leaders in Online Communities Based on An Improved PageRank Algorithm](https://www.researchgate.net/publication/272115243_Detecting_Opinion_Leaders_in_Online_Communities_Based_on_an_Improved_PageRank_Algorithm)

    - 두 노드간에 이웃 노드의 일치 정도를 edge weight으로 하여 PageRank를 변형함

* [Opinion Leader Mining Algorithm in Microblog Platform Based on Topic Similarity](https://ieeexplore.ieee.org/document/7924685/)
    
    - LDA를 통해 각 유저별로 토픽 벡터를 구함
    - 두 유저 간 토픽 벡터를 Jensen-Shannon distance를 구한 뒤 그것을 edge weight으로 사용
    - Node weight을 구하기 위한 수식을 만듦 (다만 이것을 위한 hyperparameter가 너무 많은게 단점!)


### Visualization

-   [Gatherplots: Extended Scatterplots for Categorical Data](http://www.umiacs.umd.edu/~elm/projects/gatherplots/gatherplots.pdf)

    -   categorical data 에 대해 overplotting 을 방지하면서 효율적으로 시각화 하는 방법 제시
    -   <http://www.gatherplot.org/>

*   [TopicLens: Efficient Multi-Level Topic Exploration of Large-Scale Document collections](http://www.umiacs.umd.edu/~elm/projects/topiclens/topiclens.pdf)

    -   대량의 다큐먼트에 대해 효율적으로 시각화(t-sne)하고
    -   user interaction 을 통해 각각의 토픽에 대해 다시 세부 토픽을 보여줌

-   [ReVACNN: Real-Time Visual Analytics for Convolutional Neural Network](http://poloclub.gatech.edu/idea2016/papers/p30-chung.pdf)

    -   Convolutional Neural Networks 를 효율적으로 시각화함
    -   실시간으로 학습된 필터를 시각화함으로서 학습이 잘 되는지를 알 수 있음

*   [Context Preserving Dynamic Word Cloud Visualization](http://www.shixialiu.com/publications/wordcloud/paper.pdf)

    -   워드클라우드의 각 단어들이 문맥적으로 가까울수록 가까이 위치시킬 수 있도록 정렬
    -   Importance criterion, Co-Occurrence criterion, Similarity criterion 등을 기준으로 정렬

*   [An Interactive Visual Testbed System for Dimension Reduction and Clustering of Large-scale High-dimensional Data](http://www.zcliu.org/papers/2013_vda_testbed.pdf)

    -   Large scale 데이터에 대해서 차원 축소한 뒤 효과적으로 데이터 분석 방법 제시
    -   pre-processing, clustering, dimension reduction, alignment, scatter plot 등 기능 제공
