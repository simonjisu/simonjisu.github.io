---
layout: post
title: "[deeplearning from scratch]-8: Summary"
categories: deeplearning
author: "Soo"
date: "2018-02-08 15:13:40 +0900"
comments: true
toc: true
---
# Numpy로 짜보는 Neural Network Basic - 8

---
## 총 정리
지금까지 우리는 Neural Network의 기원부터 Feedforward 과정, BackPropogation 과정, 그리고 다양한 학습 관련 기술을 배웠다. 이들을 총 정리해서 Mnist 데이터를 다시 학습 시켜보자.

[ 모든 코드 링크](https://github.com/simonjisu/NUMPYwithNN/tree/master/common)

### Package Load

```python
from common.Multilayer import MLP
from dataset.mnist import load_mnist
from common.optimizer import *
import time
```

### Data Load

```python
(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
```
>(60000, 784)<br>(60000, 10)<br>(10000, 784)<br>(10000, 10)

### Network & Optimizer settings

우리의 네트워크는 총 3층이며 Input Size가 784, Hidden node는 각각 100, 50개, Output 은 10(숫자 0~9까지의 손글씨 분류이기 때문)이다.

활성화 함수는 **ReLu**, 초기값도 이에 따라 **He** 를 써준다. 그리고 중간에 Batch Normalization을 써준다.

가중치 업데이트를 위한 옵티마이저는 **Adam** 을 쓰고, Loss Function은 **Cross Entropy** 를 쓰게 된다.

```python
nn = MLP(input_size=784, hidden_size=[100, 50], output_size=10,
         activation='relu', weight_init_std='he', use_batchnorm=True)
optimizer = Adam()
```

### Training & test

```python
train_loss_list = []
train_acc_list = []
test_acc_list = []
epoch_list = []

epoch_num=3000
train_size = x_train.shape[0]
batch_size = 100
epsilon = 1e-6

iter_per_epoch = max(train_size / batch_size, 1)

start = start = time.time()

for epoch in range(epoch_num):
    # get mini batch:
    batch_mask = np.random.choice(train_size, batch_size) # shuffle 효과
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    grads = nn.gradient(x_batch, y_batch)

    optimizer.update(nn.params, grads)

    # 1에폭당 정확도 계산
    if epoch % iter_per_epoch == 0:
        loss = nn.loss(x_batch, y_batch)
        train_loss_list.append(loss)
        train_acc = nn.accuracy(x_train, y_train)
        train_acc_list.append(train_acc)
        test_acc = nn.accuracy(x_test, y_test)
        test_acc_list.append(test_acc)
        epoch_list.append(epoch)
        print('# {0} | loss: {1:.5f} | trian acc: {2:.5f} | test acc: {3:.5f}'.format(epoch, loss, train_acc, test_acc))
    elif epoch == (epoch_num - 1):
        loss = nn.loss(x_batch, y_batch)
        train_loss_list.append(loss)
        train_acc = nn.accuracy(x_train, y_train)
        train_acc_list.append(train_acc)
        test_acc = nn.accuracy(x_test, y_test)
        test_acc_list.append(test_acc)
        epoch_list.append(epoch)
        print('# {0} | loss: {1:.5f} | trian acc: {2:.5f} | test acc: {3:.5f}'.format(epoch, loss, train_acc, test_acc))

end = time.time()
print('total time:', (end - start))        
```

>\# 0 \| loss: 11.06622 \| trian acc: 0.11408 \| test acc: 0.11910<br>\# 600 \| loss: 0.23165 \| trian acc: 0.92055 \| test acc: 0.92020<br>\# 1200 \| loss: 0.19112 \| trian acc: 0.93975 \| test acc: 0.94180<br>\# 1800 \| loss: 0.08235 \| trian acc: 0.95188 \| test acc: 0.95040<br>\# 2400 \| loss: 0.09155 \| trian acc: 0.95898 \| test acc: 0.95600<br>\# 2999 \| loss: 0.09883 \| trian acc: 0.96513 \| test acc: 0.96150<br>total time: 27.169809818267822

<img src="/assets/ML/nn/train_test-graph.png" alt="Drawing" style="width=500px"/>

시간은 3000 Epoch를 도는데 약 30초가 안걸렸으며, 테스트 결과도 우수하게 나오는 것으로 확인된다. CNN으로 하면 더 높아질 것으로 예상된다.

### Model Check

```python
def check(x, y, model):
    pred_y = model.predict(x)
    if x.ndim != 2:
        x = x.reshape(28, 28)

    print('Predict Answer: {}'.format(np.argmax(pred_y)))
    print('Real Answer: {}'.format(np.argmax(y)))
    plt.imshow(x, cmap='binary')
    plt.grid(False)
    plt.axis('off')
    plt.show()
```

테스트 데이터중 하나 골라서 실험해보자

```python
check(x_test[45], y_test[45], nn)
```
>Predict Answer: 5<br>Real Answer: 5
>
> <img src="/assets/ML/nn/num5.png" alt="Drawing" height="100" width="100"/>
