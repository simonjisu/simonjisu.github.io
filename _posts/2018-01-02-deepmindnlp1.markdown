---
layout: post
title: "DeepMindNLP 강의 정리 1"
categories: "DataScience"
author: "Soo"
date: "2018-01-02 21:39:31 +0900"
---
# Word Vectors and Lexical Semantics

## How to represent Words
* Natural language text = sequences of discrete symbols 이산 기호들의 배열(시퀀스)
* Navie representaion: one hot vectors $\in$ $R^{vocabulary}$, one hot 인코딩된 벡터들로 표현 아주큼

    words = ['딥마인드', '워드', '벡터']
    df = pd.DataFrame(np.eye(len(words)), index=words, dtype=np.int)
    df


|word|0|1|2|
|:-:|:-:|:-:|:-:|
|딥마인드|1|0|0|
|워드|0|1|0|
|벡터|0|0|1|


* Classical IR: document and query vectors are superpositions of word vectors
$$\hat{d_q}=\underset{d}{\arg \max} \sim(d,q)$$

* Similarly for word classification problems(e.g. Navie Bayes topic models)

* Issues: sparse, orthogonal representations, semantically weak

## Semantic similarity 의미론적 유사성
* 더 풍부하게 단어를 표현하고 싶다!!
* Distributional semantics: 분산 의미론
    * Idea: produce dense vector representations based on the contex/use of words
    * Approaches:
        * count-based
        * predictive
        * task-based

### Count-based methods
Define a basis vocabulary C of context words
* 고를 때는 linguistic intutition(언어적 직관, 주관적인) / statistics of the corpus 에 의해 고름
* 이것을 하는 이유는 a, the 같은 의미와 무관한 function word를 포함시키지 않기 위함

Define a word window size $w$.

Count the basis vocabulary words occurring $w$ words to the left or right of each instance of a target word in the corpus

From a vector representation of the target word based on these counts

Example:

    from collections import Counter, defaultdict
    from operator import itemgetter

    def get_vocabulary_dict(contexts, stopwords):
    vocabulary = Counter()
    for sentence in contexts:
        words = [word for word in sentence.split() if word not in stopwords]
        vocabulary.update(words)
    return vocabulary

    def represent_vector(contexts_words, vocabulary):
    vocab_len = len(vocabulary)
    word2idx = {w: i for i, w in enumerate(vocabulary)}
    count_based_vector = defaultdict()

    for key_word, context_w in contexts_words.items():
        temp = np.zeros(vocab_len, dtype=np.int)
        for w in context_w:
            temp[word2idx[w]] += 1
        count_based_vector[key_word] = temp
    return count_based_vector, word2idx

    contexts = ['and the cute kitten purred and then',
            'the cute furry cat purred and miaowed',
            'that the small kitten miaowed and she',
            'the loud furry dog ran and bit']
    stopwords=['and', 'then', 'she', 'that', 'the', 'cat', 'dog', 'kitten']
    contexts_words = {'kitten': {'cute', 'purred', 'small', 'miaowed'},
                  'cat': {'cute', 'furry', 'miaowed'},
                  'dog': {'loud', 'furry', 'ran', 'bit'}}

    vocabulary = get_vocabulary_dict(contexts, stopwords)
    count_based_vector, word2idx = represent_vector(contexts_words, vocabulary)

    word_idx_list = [w for i, w in sorted([(i, w) for w, i in word2idx.items()], key=itemgetter(0))]
    df = pd.DataFrame(count_based_vector, index=word_idx_list)
    df.T


|   |cute|purred|furry|miaowed|small|loud|ran|bit|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|cat|1|0|1|1|0|0|0|0|
|dog|0|0|1|0|0|1|1|1|
|kitten|1|1|0|1|1|0|0|0|


Compare as similarity kernel:
$cosine(u, v) = \dfrac{u\cdot v}{\|u\|\times\|v\|}$

    def cosine(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    print('kitten-cat:', cosine(df['kitten'], df['cat']))
    print('kitten-dog:',cosine(df['kitten'], df['dog']))
    print('cat-dog:',cosine(df['cat'], df['dog']))

> kitten-cat: 0.57735026919
>
> kitten-dog: 0.0
>
> cat-dog: 0.288675134595

Count-based method는 Navie Approach으로 접근

* Not all features are equal: we must distinguish counts that are high, because they are informative from those that are just independently frequent contexts.

* Many Normalisation methods: TF-IDF, PMI, etc

* Some remove the need for norm-invariant similarity metrics

But... perhaps there are easier ways to address this problem of count-based mothods(and others, e.g. choice of basis context)

### Neural Embedding Models
Learning count based vecotrs produces an embedding matrix in $R^{|vocab|\times|context|}$


|   |cute|purred|furry|miaowed|small|loud|ran|bit|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|cat|1|0|1|1|0|0|0|0|
|dog|0|0|1|0|0|1|1|1|
|kitten|1|1|0|1|1|0|0|0|


Rows are word vectors, so we can retrieve them with one hot vectors in $\{0,1\}^{|vocab|}$

$$onehot_{cat} = \begin{bmatrix} 0 \newline 1 \newline 0 \end{bmatrix}, cat=onehot_{cat}^TE$$

Symbols = unique vectors. Representation = embedding symbols with $E$


#### Generic(포괄적인) idea behind embedding learning:
1. Collect instances $t_i \in inst(t)$ of a word $t$ of vocab $V$
2. For each instance, collect its context words $c(t_i)$ (e.g. k-word window)
3. Define some score function $score(t_i, c(t_i); \theta, E)$ with upper bound on output
4. Define a loss:
$$L=-\sum_{t\in V}\sum_{t_i \in inst(t)}score(t_i, c(t_i);\theta,E)$$
5. Estimate:
$$\hat{\theta}, \hat{E}=\underset{\theta, E}{\arg \min}\ L$$
6. Use the estimated $E$ as your embedding matrix


#### Problems: Scoring function

Easy to design a useless scorer(e.g. ignore input, output upper bound)

Implicitly define is useful

Ideally, scorer:
* Embeds $t_i$ with $E$
* Produces a score which is a function of how well $t_i$ is accounted for by $c(t_i)$, and/or vice versa
* Requires the word to account for the context(or the reverse) more than another word in the same place.
* Produces a loss which is differentiable w.r.t. $\theta$ and $E$

#### C&W(Collobert et al. 2011)
[<span style="color: #7d7ee8">paper</span>](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)

Interpretation: representations carry information about what neighbouring representations should look like

where it belongs? 같은 정보를 포함

#### CBoW (Mikolov et al. 2013)
[<span style="color: #7d7ee8">paper</span>](https://arxiv.org/abs/1301.3781)

Embed context words. Add them.

Project back to vocabulary size. Softmax.
$$softmax(l)_i=\dfrac{e^{l_i}}{\sum_{j}e^{l_i}}$$
$$\begin{eqnarray} P(t_i|context(t_i) & = & softmax(\sum_{t_j\in context(t_i)} onehot_{t_j}^{t}\cdot E\cdot W_v) \newline
& = & softmax((\sum_{t_j\in context(t_i)} onehot_{t_j}^{t}\cdot E)\cdot W_v) \end{eqnarray}$$

Minimize Negative Log Likelihood:
$$L_{data} = -\sum_{t_i \in data}\log P(t_i|context(t_i))$$

장점:
* All linear, so very fast. Basically a cheap way of applying one matrix to all inputs.
* Historically, negative sampling used instead of expensive softmax.
* NLL(negative log-likelihood) minimisation is more stable and is fast enough today
* Variants: postion specific matrix per input(Ling et al. 2015)


#### Skip-gram (Mikolov et al. 2013)
[<span style="color: #7d7ee8">paper</span>](https://arxiv.org/abs/1301.3781)

Target word predicts context words.

Embed target word.

Project into vocabulary. Softmax.
$$P(t_j|t_i) = softmax(onehot_{t_i}^T\cdot E \cdot W_v)$$

Learn to estimate Likelihood of context words.
$$-\log P(context(t_i)|t_i) = -\log \prod_{t_j\in context(t_i)}P(t_j|t_i) - \sum_{t_j\in context(t_i)}\log P(t_j|t_i)$$

장점:
* Fast: One embedding versus \|C\|(size of contexts) embeddings
* Just read off probabilities from softmax
* Similiar variants to CBoW possible: position specific projections
* Trade off between efficiency and more structured notion of context

#### 기타
Word Embedding 하는 목적이 뭐냐? dense 한 vector 를 얻는 거다

Word2Vec은 딥러닝이 아니라 shallow model(얕은 모델: 층이 하나밖에 없는)이다.

Word2Vec == PMI Matrix factorization of count based models(Levy and Goldberg, 2014)

### Specific Benefits of Neural Approaches
* Easy to learn, especially with good linear algebra libraries.
* Highly parallel problem: minibatching, GPUs, distributed models.
* Can predict other discrete aspects of context(dependencies, POS tags, etc). Can estimate these probabilities with counts, but sparsity quickly becomes a problems.
* Can predict/condition on continuous contexts: e.g. images.

### Evaluating Word Representations
Intrinsic Evaluation:
* WordSim-353 (Finkelstein et al. 2003)
* SimLex-999 (Hill et al 2016, but has been around since 2014)
* Word analogy task (Mikolov et al. 2013)
* Embedding visualisation (nearest neighbours, T-SNE projection)

t-SNE visualize, word 2 dimension cluster: <span style="color: #e87d7d"> http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/ </span>

Extrinsic Evaluation:
* Simply: do your embeddings improve performance on other task(s).
* More ...

### Task-based Embedding Learning
Just saw methods for learning $E$ through minimising a loss.

One use for $E$ is to get input features to a neural network from words.

Neural network parameters are updated using gradients on loss $L(x, y, \theta)$:
$$\theta_{t+1} = update(\theta_t, \triangledown_{\theta}L(x, y, \theta_t)) $$

If $E \subseteq \theta$ then this update can modify $E$ (if we let it):
$$E_{t+1} = update(E_t, \triangledown_E L(x, y, \theta_t))$$

#### Task-based Features: Bow Classifiers
Classify sentences/documents based on a variable number of word representations

Simplest options: bag of vectors
$$P(C|D)=softmax(W_C \sum_{t_i \in D} embed_E(t_i))$$

Projection into logits (input to softmax) canbe arbitrarily complex. E.g.:
$$P(C|D)=softmax(W_C \cdot \sigma (\sum_{t_i \in D} embed_E(t_i)))$$

* $C$: class
* $D$: document

Example tasks:
* Sentiment analysis: tweets, movie reviews
* Document classification: 20 Newsgroups
* Author identification

#### Task-based Features: Bilingual Features
linguistic general approach: translations

데이터가 많으면 그냥 pre-trained할 필요 없이 Embedding을 만든(random initialize) 담에 같이 train하면 됨, 만약에 데이터가 충분치 않다면, 미리 training하는 것이 좋아 보임


## Torch로 word2vec 짜보기

    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.autograd import Variable
    import torch.utils.data as data_utils
    from torch.utils.data import DataLoader
    from scipy.spatial.distance import cosine
    import matplotlib.pylab as plt
    from collections import Counter, defaultdict, deque
    from nltk.tokenize import word_tokenize
    from operator import itemgetter

    class WORD2VEC(nn.Module):
        def __init__(self, N, half_window_size, lr, mode='cbow'):
            """
            V: vocab_size
            N: hidden layer size(word vector size)
            window_size: how many words that you want to see near target word
            mode: cbow / skipgram
            """
            super(WORD2VEC, self).__init__()

            self.V = None
            self.N = N
            # vocab and data setting
            self.half_window_size = half_window_size
            self.vocab_count = Counter()
            self.vocab2idx = defaultdict()
            self.vocab2idx['NULL'] = 0
            self.lr = lr

        def build_network(self):
            # network setting
            self.i2h = nn.Embedding(self.V, self.N, padding_idx=0)  # Embedding
            self.h2o = nn.Linear(self.N, self.V)
            self.softmax = nn.Softmax(dim=1)

        def get_vocabulary(self, corpus_list):
            for sentence in corpus_list:
                self.vocab_count.update(sentence)
            for i, w in enumerate(self.vocab_count.keys()):
                self.vocab2idx[w] = i + 1
            self.idx2vocab = {i: w for w, i in self.vocab2idx.items()}

        def generate_batch(self, sentence):
            # sentence size와 window size 결정 조건 확인(추가할것)
            target_words = []
            batch_windows = []

            # add padding data
            batch_sentence = ['NULL'] * self.half_window_size + sentence + ['NULL'] * self.half_window_size
            for i, target_word in enumerate(sentence):
                target_words.append(target_word)
                center_idx = i + self.half_window_size
                window = deque(maxlen=self.half_window_size * 2)
                window.extendleft(reversed(batch_sentence[i:center_idx]))
                window.extend(batch_sentence[center_idx + 1:center_idx + 1 + self.half_window_size])
                batch_windows.append(window)

            return batch_windows, target_words

        def data_transfer(self, corpus_list):
            """batch_data = [windows(list), target(list)]"""
            batch_data = []
            for sentence in corpus_list:
                batch_windows, target_words = self.generate_batch(sentence)
                for window, target in zip(batch_windows, target_words):
                    idxed_window = [self.vocab2idx[word] for word in window]
                    idxed_target = [self.vocab2idx[target]]
                    batch_data.append([idxed_window, idxed_target])
            return batch_data

        def tokenize_corpus(self, corpus):
            """문장에 부호를 제거하고 단어 단위로 tokenize 한다"""
            check = ['.', '!', ':', ',', '(', ')', '?', '@', '#', '[', ']', '-', '+', '=', '_']
            corpus_list = []
            for sentence in corpus:
                temp = word_tokenize(sentence)
                temp = [word.lower() for word in temp if word not in check]
                corpus_list.append(temp)
            return corpus_list

        def fit(self, corpus):
            """
            corpus를 학습시킬 데이터로 전환시켜준다. 모든 데이터는 단어의 vocab2idx를 근거해서 바뀐다.
            Vocab이 설정되면 네트워크도 같이 설정된다.
            batch_data = [window, target]
            """
            corpus_list = self.tokenize_corpus(corpus)
            self.get_vocabulary(corpus_list)
            self.V = len(self.vocab2idx)
            batch_data = self.data_transfer(corpus_list)
            self.build_network()
            print('fit done!')
            return batch_data

        def forward(self, X):
            embed = self.i2h(X)  # batch x V x N
            h = Variable(embed.data.mean(dim=1))  # batch x N
            output = self.h2o(h)  # batch x V
            probs = self.softmax(output)  # batch x V
            return output, probs

    #################################################
    # Sample Data
    #################################################

    def create_sample_data():
        corpus = ['the king loves the queen',
                  'the queen loves the king',
                  'the dwarf hates the king',
                  'the queen hates the dwarf',
                  'the dwarf poisons the king',
                  'the dwarf poisons the queen',]

        return corpus


    def get_data_loader(batch_data, batch_size, num_workers, shuffle=False):
        features = torch.LongTensor([batch_data[i][0] for i in range(len(batch_data))])
        targets = torch.LongTensor([batch_data[i][1] for i in range(len(batch_data))])
        data = data_utils.TensorDataset(features, targets)

        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return loader

    #################################################
    # Train
    #################################################

    def word2vec_train(corpus, N, half_window_size=2, lr=0.01, n_epoch=1000, batch_size=10, print_epoch=100, num_workers=2, shuffle=False):
        """본격적으로 데이터를 학습한다"""
        word2vec = WORD2VEC(N=N, half_window_size=half_window_size, lr=lr)
        batch_data = word2vec.fit(corpus)
        loader = get_data_loader(batch_data, batch_size, num_workers, shuffle)

        F = nn.CrossEntropyLoss()
        optimizer = optim.SGD(word2vec.parameters(), lr=word2vec.lr)

        loss_list = []
        for epoch in range(n_epoch):

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                batch_X = Variable(batch_X)
                batch_y = Variable(batch_y)

                output, probs = word2vec.forward(batch_X)
                loss = F(output, batch_y.squeeze(-1))  # must be 1-d tensor in labels

                loss.backward()
                optimizer.step()
            loss_list.append(loss.data[0])

            if epoch % print_epoch == 0:
                print('#{}| loss:{}'.format(epoch, loss.data[0]))

        return word2vec, loss_list

Training은 아래와 같다.

    corpus = create_sample_data()
    N = 2
    half_window_size = 2
    lr = 0.01
    n_epoch = 3000
    print_epoch = 200
    batch_size = 4
    num_workers=2
    shuffle=False
    word2vec, loss_list = word2vec_train(corpus, N=N, half_window_size=half_window_size,
      lr=lr, n_epoch=n_epoch, batch_size=batch_size, print_epoch=print_epoch,
      num_workers=num_workers, shuffle=shuffle)


<img src="/assets/ML/Deepnlp/lec1/Loss.png" alt="Drawing" style="width: 300px;"/>

2차원으로 embedding 했으니 평면에 그려보았다.

<img src="/assets/ML/Deepnlp/lec1/vector.png" alt="Drawing" style="width: 300px;"/>

조금더 큰 데이터를 그냥 CBOW 혹은 Skip-gram으로 학습 시킬 경우 속도가 아주 느린 것을 발견 할 수 가 있다. 이는 말뭉치가 많아질 수록 단어의 수도 많아 지기 때문에, 말단에 Hierarchical Softmax와 Negative Sampling 방법을 쓴다고 한다. 김범수님의 블로그 [[<span style="color: #7d7ee8">링크</span>]](https://shuuki4.wordpress.com/2016/01/27/word2vec-%EA%B4%80%EB%A0%A8-%EC%9D%B4%EB%A1%A0-%EC%A0%95%EB%A6%AC/) 참조
