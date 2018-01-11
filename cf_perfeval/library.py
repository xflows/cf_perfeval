def perfeval_classification_statistics(input_dict):
    from sklearn import metrics
    labels = input_dict['true_and_predicted_labels']
    pos_label = input_dict.get('pos_label', None)
    # Check if we have true and predicted labels for each fold
    if labels and type(labels[0][0]) == list:
        try:
            # Flatten
            y_true, y_pred = [], []
            for fold_labels in labels:
                y_true.extend(fold_labels[0])
                y_pred.extend(fold_labels[1])
            labels = [y_true, y_pred]
        except:
            raise Exception('Expected true and predicted labels for each fold, but failed.' +
                            'If you wish to provide labels for each fold separately it should look like: ' +
                            '[[y_true_1, y_predicted_1], [y_true_2, y_predicted_2], ...]')
    if len(labels) != 2:
        raise Exception('Wrong input structure, this widget accepts labels in the form: [y_true, y_pred]')

    y_true, y_pred = labels

    classes = set()
    classes.update(y_true + y_pred)
    classes = sorted(list(classes))

    # Assign integers to classes
    class_to_int = {}
    for i, cls_label in enumerate(classes):
        class_to_int[cls_label] = i

    y_true = [class_to_int[lbl] for lbl in y_true]
    y_pred = [class_to_int[lbl] for lbl in y_pred]

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    # AUC is defined only for binary classes
    if len(classes) == 2:
        auc = metrics.roc_auc_score(y_true, y_pred)
    else:
        auc = 'undefined for multiple classes'
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'auc': auc, 'confusion_matrix': confusion_matrix}


def perfeval_noise_detection(input_dict):
    noise = input_dict['noisy_inds']
    nds = input_dict['detected_noise']

    performance = []
    for nd in nds:
        nd_alg = nd['name']
        det_noise = nd['inds']
        inboth = set(noise).intersection(set(det_noise))
        recall = len(inboth) * 1.0 / len(noise) if len(noise) > 0 else 0
        precision = len(inboth) * 1.0 / len(det_noise) if len(det_noise) > 0 else 0
        beta = float(input_dict['f_beta'])
        print(beta, recall, precision)
        if precision == 0 and recall == 0:
            fscore = 0
        else:
            fscore = (1 + beta ** 2) * precision * recall / ((beta ** 2) * precision + recall)
        performance.append({'name': nd_alg, 'recall': recall, 'precision': precision, 'fscore': fscore, 'fbeta': beta})

    from operator import itemgetter
    output_dict = {}
    output_dict['nd_eval'] = sorted(performance, key=itemgetter('name'))
    return output_dict


def perfeval_bar_chart(input_dict):
    return {}


def perfeval_to_table(input_dict):
    return {}


def perfeval_batch(input_dict):
    alg_perfs = input_dict['perfs']
    beta = float(input_dict['beta'])
    performances = []
    for exper in alg_perfs:
        noise = exper['positives']
        nds = exper['by_alg']

        performance = []
        for nd in nds:
            nd_alg = nd['name']
            det_noise = nd['inds']
            inboth = set(noise).intersection(set(det_noise))
            recall = len(inboth) * 1.0 / len(noise) if len(noise) > 0 else 0
            precision = len(inboth) * 1.0 / len(det_noise) if len(det_noise) > 0 else 0

            print(beta, recall, precision)
            if precision == 0 and recall == 0:
                fscore = 0
            else:
                fscore = (1 + beta ** 2) * precision * recall / ((beta ** 2) * precision + recall)
            performance.append(
                {'name': nd_alg, 'recall': recall, 'precision': precision, 'fscore': fscore, 'fbeta': beta})

        performances.append(performance)

    output_dict = {}
    output_dict['perf_results'] = performances
    return output_dict


def perfeval_aggr_results(input_dict):
    output_dict = {}
    output_dict['aggr_dict'] = {'positives': input_dict['pos_inds'], 'by_alg': input_dict['detected_inds']}
    return output_dict


def perfeval_pr_space(input_dict):
    return {}
