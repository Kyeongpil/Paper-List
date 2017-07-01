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
