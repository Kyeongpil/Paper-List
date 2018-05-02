# DataMining & Visualization



### DataMining

* [Deep content-based music recommendation](https://papers.nips.cc/paper/5004-deep-content-based-music-recommendation)

    - CNN을 통해 음악 feature를 extract
    - Weighted Matrix Factorization을 통해 추천


* [Collaborative Feature Learning from Social Media](https://arxiv.org/abs/1502.01423)

    - 이미지는 CNN을 통해 feature를 뽑음
    - Matrix factorization(with Negative sampling)을 통해 Collaborative filtering 수행


* [Learning Topics in Short Texts by Non-negative Matrix Factorization on Term Correlation Matrix](http://epubs.siam.org/doi/abs/10.1137/1.9781611972832.83)

    - term occurrence를 PPMI(Positive Pointwise Mutual Information) matrix 만들고
    - NMF(Non-negative matrix factorization)을 통해 토픽 모델링 수행


* [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)

    - Neural Network에 deep 하게 쌓은 network와 wide 한 network 결합
    - 딥러닝을 추천 시스템에 사용
    - Training 크기 500 Billion ...


* [Learning Entity and Relation Embeddings for Knowledge Graph Completion](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/aaai2015_transr.pdf)

    - Knowlege graph embedding 모델로 TransR 제안
    - entity space와 relation space를 같은 공간에 embedding함
    - relation을 클러스터링하여 관계를 그룹핑한 CTransR 버전


* [Knowledge Graph Embedding for Hyper-Relational Data](http://ieeexplore.ieee.org/document/7889640/)

    - Hyper-relation(두 entity간 다중의 relation이 있을 수 있음) 문제를 해결하기 위한 모델 제안
    - transHR, hyper-relation에 대해 각각의 relation matrix를 통해 embedding함


* [TransG: A Generative Model for Knowledge Graph Embedding](https://aclweb.org/anthology/P/P16/P16-1219.pdf)

    - multiple relation 문제 -> relation에 클러스터링 도입
    - 클러스터링 갯수를 모름 -> Chinese restaurant process로 동적 생성


* [Learning to Extract Conditional Knowledge for Question Answering Using Dialogue](http://dl.acm.org/citation.cfm?id=2983777)

    - 대화를 할 때 conditional knowledge base(CKB)를 추출
    - clustering을 통해 answer를 할 때 해당 CKB와 관련이 있는 답을 하도록 학습


* [Augmented Variational Autoencoders for Collaborative Filtering with Auxiliary Information](https://dl.acm.org/citation.cfm?id=3132972)

    - 추천시스템에 Variational autoencoder와 Adversarial regularization을 융합한 모델 제안
    - 이외에 기존의 여러 종류의 VAE 모델도 적용


* [Latent Relational Metric Learning via Memory-based Attention for Collaborative Ranking](https://www.researchgate.net/publication/322385353_Latent_Relational_Metric_Learning_via_Memory-based_Attention_for_Collaborative_Ranking)

    - 추천시스템에 Knowledge graph embedding 기법과 딥러닝의 attention(memory) mechanism 도입
    - 각 user와 item 간의 relation vector 도입


* [The spread true and false news online](http://science.sciencemag.org/content/359/6380/1146)

    - Fake news detection 관련
    - 여러 fake news detection site에서 fake, true 뉴스 링크를 통해 트위터에서 해당 링크를 공유하는 트윗을 수집
    - 해당 트윗들을 리트윗(spread, cascade)하는 범위(depth, breadth) 및 속도, 갯수를 조사
    - fake news가 true news보다 더 깊고 넓게 확산됨
    - 그 이유를 몇가지 연구를 통해 분석
    - 막상 fake news를 퍼뜨리는 user는 follower 수가 그리 많지는 않음
    - fake news들은 대체적으로 novelty attract가 높음
    - 감정분석을 한 결과 fake news와 true news가 갖는 감정이 대체적으로 다름


* [What is Twitter, a Social Network or a News Media?](https://dl.acm.org/citation.cfm?id=1772751)

    - 트위터에서의 social network 분석
    - follow, retweet 관계 등에 따른 user 영향도 등 분석
    - 토픽의 트렌드가 어떻게 퍼지고 지속되는지도 분석함



### Visualization

* [Gatherplots: Extended Scatterplots for Categorical Data](http://www.umiacs.umd.edu/~elm/projects/gatherplots/gatherplots.pdf)

    - categorical data에 대해 overplotting을 방지하면서 효율적으로 시각화 하는 방법 제시
    - <http://www.gatherplot.org/>


* [TopicLens: Efficient Multi-Level Topic Exploration of Large-Scale Document collections](http://www.umiacs.umd.edu/~elm/projects/topiclens/topiclens.pdf)

    - 대량의 다큐먼트에 대해 효율적으로 시각화(t-sne)하고
    - user interaction을 통해 각각의 토픽에 대해 다시 세부 토픽을 보여줌


* [ReVACNN: Real-Time Visual Analytics for Convolutional Neural Network](http://poloclub.gatech.edu/idea2016/papers/p30-chung.pdf)

    - Convolutional Neural Networks를 효율적으로 시각화함
    - 실시간으로 학습된 필터를 시각화함으로서 학습이 잘 되는지를 알 수 있음


* [Context Preserving Dynamic Word Cloud Visualization](http://www.shixialiu.com/publications/wordcloud/paper.pdf)
    - 워드클라우드의 각 단어들이 문맥적으로 가까울수록 가까이 위치시킬 수 있도록 정렬
    - Importance criterion, Co-Occurrence criterion, Similarity criterion 등을 기준으로 정렬

* [An Interactive Visual Testbed System for Dimension Reduction and Clustering of Large-scale High-dimensional Data](http://www.zcliu.org/papers/2013_vda_testbed.pdf)

    - Large scale 데이터에 대해서 차원 축소한 뒤 효과적으로 데이터 분석 방법 제시
    - pre-processing, clustering, dimension reduction, alignment, scatter plot 등 기능 제공
