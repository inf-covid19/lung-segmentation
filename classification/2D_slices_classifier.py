import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


CSV_FILE = 'all_preds_raw.csv'

# LABELS = ['covid', 'non covid']
LABELS = ['covid', 'others']


def get_cm_metrics(confusion_matrix, labels, verbose=False):
    n = len(labels)
    cm_total = confusion_matrix.sum()
    chance_agree = 0
    agree = confusion_matrix.trace() / cm_total

    supports = [0] * n
    vp = [0] * n
    fp = [0] * n
    fn = [0] * n
    precisions = [0] * n
    recalls = [0] * n
    fscores = [0] * n

    label_true_sum = [0] * n
    label_pred_sum = [0] * n

    for row in range(n):
        for column in range(n):
            value = confusion_matrix[row][column]

            supports[row] += value

            label_true_sum[row] += value
            label_pred_sum[column] += value

            if row == column:
                vp[row] += value
            else:
                fp[column] += value
                fn[row] += value

    for x in range(n):
        recalls[x] = vp[x] / (vp[x] + fn[x])
        precisions[x] = vp[x] / (vp[x] + fp[x])

        if precisions[x] == 0 or recalls[x] == 0:
            fscores[x] = 0
        else:
            fscores[x] = 2 * (precisions[x] * recalls[x]) / \
                (precisions[x] + recalls[x])

        prob_label_true = label_true_sum[x] / cm_total
        prob_label_pred = label_pred_sum[x] / cm_total

        chance_agree += prob_label_true * prob_label_pred

    if verbose:
        print()
        print(''.rjust(30), 'precision'.rjust(10), 'recall'.rjust(
            10), 'f1-score'.rjust(10), 'support'.rjust(10))
        print()
        for x in range(n):
            print(labels[x].ljust(30),
                  '{:.2f}'.format(precisions[x]).rjust(10),
                  '{:.2f}'.format(recalls[x]).rjust(10),
                  '{:.2f}'.format(fscores[x]).rjust(10),
                  '{}'.format(supports[x]).rjust(10))
        print()
    accuracy = sum(vp) / sum(supports)
    avg_fscore = sum(fscores) / n
    kappa_score = (agree - chance_agree) / (1 - chance_agree)

    return {
        'F1_scores': fscores,
        'accuracy': accuracy,
        'macro_F1': avg_fscore,
        'supports': supports,
        'precisions': precisions,
        'recalls': recalls,
        'kappa': kappa_score,
    }


def plot_confusion_matrix(cm,
                          labels,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False,
                          normalize_color=True,
                          show_zero=True,
                          clear_diagonal=False,
                          output_file=None,
                          simple=False,
                          figsize=(16, 12),
                          verbose=False):

    metrics = get_cm_metrics(cm, labels, verbose)
    supports = metrics['supports']
    labels = [f"{labels[i]} ({supports[i]})" for i in range(len(labels))]

    accuracy = np.trace(cm) / float(np.sum(cm))
    kappa = metrics['kappa']

    macro_f1 = metrics['macro_F1']

    if clear_diagonal:
        for i in range(cm.shape[0]):
            cm[i, i] = 0

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig_x = max(len(labels) // 10, 1)
    figsize = (figsize[0] * fig_x, figsize[1] * fig_x)

    plt.figure(figsize=figsize)

    matrix = cm
    if normalize_color:
        matrix = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(matrix, interpolation="nearest", cmap=cmap)
    plt.title(title)

    if labels is not None:
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45, ha='right')
        plt.yticks(tick_marks, labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = matrix.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if show_zero or cm[i, j] > 0:
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if matrix[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}\n{:0.1f}%".format(cm[i, j], cm_normalized[i, j] * 100),
                         horizontalalignment="center",
                         verticalalignment="center",
                         color="white" if matrix[i, j] > thresh else "black")

    if not simple:
        ax = plt.gca()
        plt.tick_params(axis='x', labelbottom=True,
                        labeltop=True, bottom=True, top=True)
        plt.tick_params(axis='y', labelright=True)

        # Rotate and align top ticklabels
        plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
                 ha="left", va="center", rotation_mode="anchor")

    plt.margins(1)
    plt.ylabel('True label')
    plt.xlabel('Predicted label\n' +
               'accuracy={:0.4f}\n'.format(accuracy) +
               'macro-f1={:0.4f}\n'.format(macro_f1) +
               'kappa={:0.4f}\n'.format(kappa))
    plt.rcParams['figure.facecolor'] = 'white'

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f'Saved plot result to {output_file}')
    else:
        plt.show()


def main():

    df = pd.read_csv(CSV_FILE, index_col=0)

    exams_labels = df.columns.tolist()

    errors = []

    print("df")
    print(df)
    print()

    values = df.iloc[:-1].values
    X = []

    for r in values:
        X.append([v.replace(',', '.') for v in r])
    X = np.array(X).astype(float)
    X = X.transpose()
    print('X', X.shape)

    labels = df.iloc[-1].values
    Y = []

    for r in labels:
        y = [0, 0]
        y[LABELS.index(r)] = 1

        Y.append(y)
    Y = np.array(Y).astype(int)

    print('Y', Y.shape)

    folds = []

    total_idx = list(range(len(X)))

    random.shuffle(total_idx)

    fold_size = len(X) // 5

    for i in range(5):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size

        test_fold = total_idx[i * fold_size: end_idx]

        training_fold = list(set(total_idx) - set(test_fold))

        folds.append((training_fold, test_fold))

    final_matrix = np.zeros((2, 2)).astype(int)

    for fold in folds:
        training_idx, test_idx = fold

        x_train = X[training_idx]
        y_train = Y[training_idx]
        y_train = np.array([np.argmax(y) for y in y_train])

        x_test = X[test_idx]
        y_test = Y[test_idx]
        y_test = np.array([np.argmax(y) for y in y_test])

        # TODO: Add different models
        # model = LogisticRegression(max_iter=1000).fit(x_train, y_train)

        model = RandomForestClassifier().fit(x_train, y_train)

        y_pred = model.predict(x_test)

        matrix = confusion_matrix(y_test, y_pred)

        print('confusion matrix')
        print(matrix)
        print()

        training_acc = model.score(x_train, y_train)
        test_acc = model.score(x_test, y_test)

        # TODO: Show more metrics: f1-score, kappa
        print('training acc', training_acc)
        print('test acc', test_acc)
        print()

        final_matrix += matrix

        print('test idx', test_idx)

        # Loading errors
        for i in range(len(y_test)):
            if y_test[i] != y_pred[i]:
                exam_label = exams_labels[test_idx[i]]
                errors.append((exam_label, y_test[i], y_pred[i]))

    print()
    print("List of errors:")
    for r in errors:
        print(f"Exam: {r[0].ljust(10)} => true: {r[1]} | pred: {r[2]}")
    print()

    print('final matrix')
    print(final_matrix)

    cm_filename = f"{CSV_FILE.replace('.csv', '')}-confusion-matrix.png"
    cm_title = f"Confusion Matrix: {CSV_FILE}"

    plot_confusion_matrix(final_matrix, LABELS, title=cm_title, output_file=cm_filename,
                          figsize=(6, 6))


if __name__ == "__main__":
    # TODO: Add arguments:
    #       - input files
    #       - output folder to png files
    #       - verbose flag
    #       - k to repeat k cross validations
    main()
