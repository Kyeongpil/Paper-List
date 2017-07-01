# Deep Learning Papers


### Vision & Video

* [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)

    - 첫 GAN 모델 제안
    - Minimax game


* [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328)

    - Generative Model
    - Gated convolutional layers, residual connections
    - latent vector h를 추가함으로서 conditional Generative model로 사용 가능


### Word/Sentence Embedding

* [Query Expansion with Locally-Trained Word Embeddings](http://www.aclweb.org/anthology/P16-1035)

    - Local word Embedding 개념을 도입


* [Cross-lingual Models of Word Embeddings: An Empirical Comparison](http://www.aclweb.org/anthology/P16-1157)

    - Cross lingual word embedding 4가지 모델(BiSkip, BiCVM, BiCCA, BiVCD)를 실험적으로 비교함


* [Embeddings for Word Sense Disambiguation: An Evaluation Study](http://www.aclweb.org/anthology/P16-1085)

    - Word Sense Disambiguation를 해결하기 위해 몇가지 기법들을 제시
    - 이 중 Fractional decay가 효과적 (이웃 단어들의 거리를 고려한 가중 평균)


* [On the Role of Seed Lexicons in Learning Bilingual Word Embeddings](http://www.aclweb.org/anthology/P16-1024)

    - 기본적인 seed lexicon들을 가지고 bilingual에 대해 어떻게 학습을 할지 실험 비교


* [Distributed Representations of Words and Phrases and their Compositionality](http://web2.cs.columbia.edu/~blei/seminar/2016_discrete_data/readings/MikolovSutskeverChenCorradoDean2013.pdf)

    - Skip-gram 모델 첫 제안


* [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726)

    - Sentence를 embedding 하는 방법 제안
    - 주어진 input sentence를 RNN Encoder-Decoder에 넣어 전/후 sentence를 예측하도록 학습


* [Swivel: Improving Embeddings by Noticing Whats Missing](https://arxiv.org/abs/1602.02215)

    - 전체 PMI를 이용하여 word embedding 학습
    - PMI Matrix가 매우 크므로 분산 처리
    - Matrix를 block으로 나눠서 분산 처리 함 


### Natural Language Processing, Language Model, QA

* [Sequence to Sequence Learning with Neural Netowrks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

    - Encoder Decoder 형식의 Sequence to sequence model 제안
    - Encoder의 마지막 hidden vector를 sentence를 embedding할 수 있음을 시각화함


* [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Neural Traslation](https://arxiv.org/abs/1406.1078)

    - GRU 모델 제안, GRU 기반의 sequence to sequence 모델


* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

    - 첫 Attention Mechanism 제안
    - Attention을 통해 학습이 잘되고 더 긴 문장에 대해서도 효과적


* [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473)

    - Generative Adversarial Networks를 sequence 모델에 도입한 거의 첫 사례
    - Policy gradient와 MC sampling을 사용하여 학습


* [Neural Networks For Negation Scope Detection](http://www.aclweb.org/anthology/P16-1047)

    - 부정어의 범위를 찾기 위해 뉴럴넷 사용


* [Event Detection and Domain Adaptation with Convolutional Neural Networks](http://anthology.aclweb.org/P/P15/P15-2060.pdf)

    - Word Embedding과 CNN을 사용하여 text에서 event detection 수행


* [Using Sentence-Level LSTM Language Models for Script Inference](https://arxiv.org/abs/1604.02993)

    - Rare Word에 대한 embedding을 common word의 조합으로 대체
    - ZRegression으로 softmax를 대체하여 효율적으로 처리


* [Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181)

    - word 단위로 embedding한 벡터들의 sequence를 CNN과 max pooling을 통해 text classification 수행
    - 기존 RNN에 비해 classification accuracy도 좋고 학습 속도도 빠름


* [Neural Summarization by Extracting Sentence and Words](https://arxiv.org/abs/1603.07252)

    - Encode part에서 recurrent convolutional document reader 사용
    - word extraction에서는 neural attention 모델 사용


* [Sentence Compression by Deletion with LSTMs](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/43852.pdf)

    - LSTM 모델을 이용한 sentence compression(summarization)


* [Learning to Compose Neural Networks for Question Answering](https://arxiv.org/abs/1601.01705)


* [End-To-End Memory Networks](https://arxiv.org/abs/1503.08895)

    - Memory network(multiple layer)를 도입하여 input sequences와 question을 비교하며 어느 부분을 단계적으로 주목해야할지 계산
    - Question Answering에서 유용


* [Generating Factoid Questions With Recurrent Neural Networks: The 30M Factoid Question-Answer Corpus](http://www.aclweb.org/anthology/P16-1056)

    - GRU 모델을 이용, Freebase 데이터셋을 기초로 하여 QA corpus를 생성함


* [Ask Me Anything: Dynamic Memory Neural Networks for Natural Language Processing](https://arxiv.org/abs/1506.07285)

    - GRU, Attention model을 이용한 Dynamic Memory Networks


* [Abstractive Sentence Summarization with Attentive Recurrent Neural Networks](http://nlp.seas.harvard.edu/papers/naacl16_summary.pdf)

    - RNN 모델에 Attention mechanism을 적용하여 sentence summarization을 수행
    - 흥미로운 점은 Elman RNN(기존 RNN)이 LSTM보다 결과가 좋음 - overfitting일수도...


* [Text Understanding with the Attention Sum Reader Network](https://arxiv.org/abs/1603.01547)

    - cloze test (빈칸추론)을 수행하기 위해 bidirectional RNN 사용
    - 여기에 attention sum reader 모델(Pointer network에 영감) 소개


* [Towards AI-Complete Question Answering: A Set of Prerequesite Toy Tasks](https://arxiv.org/abs/1502.05698)

    - bAbI를 생성
    - 기존 모델 - LSTM, SVM, MemNN 등을 실험을 통해 비교



### Speech & Sound

* [Towards End-to-End Speech Recognition with Recurrent Neural Networks](http://mickey-luke.de/ASR2.pdf)
    
    - 음성을 텍스트로 인식하는 task
    - Bidirectional RNN 사용
    - 음성 길이와 텍스트 길이가 다르므로 학습을 위해 Connectionist Temporal Classification(CTC) 방법 제시


* [WaveNet: A Generative Model For Raw Audio](https://arxiv.org/abs/1609.03499)

    - Directed Casual Convolutions 모델 제안
    - Gated Activation units, residual, skip connections 사용
    - TTS(Text to Speech)에도 적용





### Deep learning theory & Model

* [Training Products of Experts by Minimizing Constrative Divergence](http://www.cs.toronto.edu/~fritz/absps/tr00-004.pdf)

    - RBM을 효율적으로 학습하기 위해 Gibbs sampling과 Constrative Divergence 방법 도입?


* [A fast learning algorithm for deep belief nets](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)

    - Generative model인 DBN을 효율적으로 학습하기 위한 방법 제시?


* [Maxout networks](http://proceedings.mlr.press/v28/goodfellow13.pdf)

    - 새로운 형태의 activation function(maxout) 제시


* [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)

    - 모델의 크기를 줄이면서 성능은 거의 유지시킴
    - 1 by 1 filter 도입
