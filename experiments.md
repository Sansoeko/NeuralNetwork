####Experiments

#####Average
- model average score on the valid data set: 
```
[0.082, 0.1063, 0.0496, 0.0632, 0.0669, 0.0299, 0.0755, 0.0863, 0.0409, 0.0718]
67.24%!
```
As 07-09-2017
- model average (absb() changed to **2)
             precision    recall  f1-score   support

          0       0.92      0.87      0.90       991
          1       0.77      0.97      0.86      1064
          2       0.89      0.80      0.84       990
          3       0.77      0.80      0.79      1030
          4       0.83      0.82      0.83       983
          5       0.77      0.70      0.73       915
          6       0.91      0.90      0.91       967
          7       0.92      0.86      0.89      1090
          8       0.82      0.76      0.79      1009
          9       0.74      0.80      0.77       961

avg / total       0.83      0.83      0.83     10000

Accuracy: 83.07%!

As 11-09-2017

#####Contrast
- model contrast score on the valid data set:

Why is the score lower than the from Average? 
```
[0.082, 0.1063, 0.0496, 0.0632, 0.0669, 0.0299, 0.0755, 0.0863, 0.0409, 0.0718]
62.17%!
```
As 07-09-2017


#####neural net with one hidden layer
```
Learning rates: 0.1
With hidden layers: 1
With neurons per hidden layer: 80
Times learned: 17
             precision    recall  f1-score   support

          0       0.97      0.98      0.98       991
          1       0.99      0.99      0.99      1064
          2       0.97      0.97      0.97       990
          3       0.93      0.97      0.95      1030
          4       0.98      0.96      0.97       983
          5       0.98      0.94      0.96       915
          6       0.97      0.99      0.98       967
          7       0.98      0.97      0.98      1090
          8       0.97      0.97      0.97      1009
          9       0.96      0.95      0.96       961

avg / total       0.97      0.97      0.97     10000

Accuracy: 97.04%!
On the testset:
             precision    recall  f1-score   support

          0       0.98      0.98      0.98       980
          1       0.99      0.99      0.99      1135
          2       0.97      0.97      0.97      1032
          3       0.94      0.99      0.96      1010
          4       0.98      0.97      0.97       982
          5       0.98      0.96      0.97       892
          6       0.97      0.98      0.97       958
          7       0.97      0.96      0.96      1028
          8       0.97      0.96      0.96       974
          9       0.97      0.94      0.96      1009

avg / total       0.97      0.97      0.97     10000

Accuracy on testset: 97.08%!
```
As 04-10-2017

```
Learning rates: 0.1
With hidden layers: 1
With neurons per hidden layer: 90
Times learned: 10
             precision    recall  f1-score   support

          0       0.98      0.98      0.98       991
          1       0.99      0.99      0.99      1064
          2       0.98      0.96      0.97       990
          3       0.93      0.98      0.96      1030
          4       0.98      0.97      0.97       983
          5       0.98      0.94      0.96       915
          6       0.96      0.99      0.98       967
          7       0.98      0.98      0.98      1090
          8       0.96      0.97      0.97      1009
          9       0.96      0.95      0.95       961

avg / total       0.97      0.97      0.97     10000

Accuracy: 97.11%!
On the testset:
             precision    recall  f1-score   support

          0       0.97      0.99      0.98       980
          1       0.99      0.98      0.98      1135
          2       0.97      0.96      0.97      1032
          3       0.93      0.98      0.96      1010
          4       0.98      0.97      0.97       982
          5       0.98      0.95      0.97       892
          6       0.97      0.99      0.98       958
          7       0.97      0.96      0.97      1028
          8       0.96      0.96      0.96       974
          9       0.97      0.96      0.96      1009

avg / total       0.97      0.97      0.97     10000

Accuracy on testset: 96.99%!

```
as 05-10-2017