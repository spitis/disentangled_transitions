import numpy as np
from sklearn import metrics


def derive_acc(fpr, tpr, labels):
  """Compute accuracy per threshold given false/true positive rates and labels.

  Cf. https://en.wikipedia.org/wiki/Precision_and_recall
  """
  p = labels.sum()  # total num positive labels
  n = (1. - labels).sum()  # total num negative labels
  tp = p * tpr  # total num true positives
  tnr = 1. - fpr  # true negative rate
  tn = n * tnr  # total num false positives
  acc = (tp  + tn) / (p + n)  # accuracy
  return acc


def derive_f1(fpr, tpr, labels):
  """Compute F1 per threshold given false/true positive rates and labels.

  Cf. https://en.wikipedia.org/wiki/Precision_and_recall
  """
  p = labels.sum()  # total num positive labels
  n = (1. - labels).sum()  # total num negative labels
  tp = p * tpr  # total num true positives
  fp = n * fpr  # total num true positives
  ppv = tp / (tp + fp)  # positive predictive value
  f1 = 2 * (ppv * tpr) / (ppv + tpr)  # f1 score
  f1[np.isnan(f1)] = 0.  # handle numerical corner case
  return f1


if __name__ == '__main__':
  num_examples = 10000
  labels = np.random.binomial(1, p=.3, size=(num_examples, ))
  scores = labels + np.random.randn(*labels.shape)
  scores = scores.clip(0., 1.)
  fpr, tpr, thresh = metrics.roc_curve(labels, scores)

  acc = derive_acc(fpr, tpr, labels)
  for acc_, thresh_ in zip(acc, thresh):
    preds = (scores >= thresh_).astype(scores.dtype)
    _acc = metrics.accuracy_score(labels, preds)
    assert np.isclose(acc_, _acc, atol=1e-3), \
      'bad accuracy at thresh {}. got {}, expected {}'.format(
        thresh_, acc_, _acc)

  f1 = derive_f1(fpr, tpr, labels)
  for f1_, thresh_ in zip(f1, thresh):
    preds = (scores >= thresh_).astype(scores.dtype)
    _f1 = metrics.f1_score(labels, preds)
    assert np.isclose(f1_, _f1, atol=1e-3), \
      'bad f1 at thresh {}. got {}, expected {}'.format(
        thresh_, f1_, _f1)

  print('test passed')
