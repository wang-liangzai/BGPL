import argparse
import os

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_data_path", default="./outputs/label", type=str)
    parser.add_argument("--oas_data_path", default="./outputs/oas", type=str)
    parser.add_argument("--aos_data_path", default="./outputs/aos", type=str)
    parser.add_argument("--oas_prob_path", default="./outputs/oas/temp", type=str)
    parser.add_argument("--aos_prob_path", default="./outputs/aos/temp", type=str)
    parser.add_argument("--conflict_strategy", default=True)
    parser.add_argument("--integrate_strategy", default=True)


    parser.add_argument("--do_train",
                        default=False,
                        help="Whether to run training.")
    parser.add_argument(
        "--do_inference",
        default=True,
        help="Whether to run inference with trained checkpoints")
    args = parser.parse_args()
    return args

def fusion(args):
    fall_aspect, fall_opinion, fall_pair, fall_apce, fpred_aspect, fpred_opinion, fpred_pair, fpred_apce = [], [], [], [], [], [], [], []

    labels, oas, aos, oas_prob, aos_prob, result_pre, result_label = [], [], [], [], [], [], []
    n_gold, n_pred, n_tp, top_f1, top_precision, top_recall, pred, tp = 0, 0, 0, 0, 0, 0, 0, 0
    result_path = os.path.join(args.label_data_path, "result_label.txt")
    aos_pred_path = os.path.join(args.oas_data_path, "result_oas.txt")
    oas_pred_path = os.path.join(args.aos_data_path, "result_aos.txt")
    aos_prob_path = os.path.join(args.oas_prob_path, "prob_file.txt")
    oas_prob_path = os.path.join(args.aos_prob_path, "prob_file.txt")

    with open(result_path, 'r') as file1:
        for line in file1:
            line = line.lower()
            labels.append(eval(line))
    with open(oas_pred_path, 'r') as file2:
        for line in file2:
            line = line.lower()
            oas.append(eval(line))
    with open(aos_pred_path, 'r') as file3:
        for line in file3: 
            line = line.lower()
            aos.append(eval(line))
    with open(oas_prob_path, 'r') as file4:
        for line in file4:
            line = line.lower()
            oas_prob.append(eval(line))
    with open(aos_prob_path, 'r') as file5:
        for line in file5: 
            line = line.lower()
            aos_prob.append(eval(line))

    b = 0.91
    if args.conflict_strategy:
        for j in range(len(aos)):
            for item_a in aos[j]:
                prob_num = 0
                for item_o in oas[j]:
                    prob_num += 1
                    if item_o[3] == item_a[3] and item_a[2] == item_o[2]:
                        if item_o[1] != item_a[1]:    
                            if item_a[1] in item_o[1]:
                                if item_a[3] == item_o[3]:    
                                    if len(aos_prob[j]) != 0 and len(oas_prob[j]) != 0:
                                        pro_1 = aos_prob[j][0]
                                        if b*float(pro_1[1:-1]) >= float(oas_prob[j][(prob_num-1)*3+0][1:-1]) and len(item_a) < len(item_o):
                                            pass
                                        else:
                                            aos[j].remove(item_a)
                                            aos[j].append(item_o)
                            if item_o[1] in item_a[1]: 
                                if item_a[3] == item_o[3]:    
                                    if len(aos_prob[j]) != 0 and len(oas_prob[j]) != 0 and item_a[2] == item_o[2] :
                                        pro_2 = oas_prob[j][(prob_num-1)*3+0]
                                        if float(aos_prob[j][0][1:-1]) >= b*float(pro_2[1:-1]) and len(item_o) < len(item_a):
                                            pass
                                        else:
                                            oas[j].append(item_a)
                                            oas[j].remove(item_o)


    if args.integrate_strategy:
        for i in range(len(aos)):
            for j in oas[i]:
                if len(oas_prob[i]) != 0:
                    values = [float(item.strip('[]')) for item in oas_prob[i]]
                    if j not in aos[i] and values[0] > 0.97 and values[0] < 0.9996 and values[1] > 0.98 and values[1] < 0.999 and values[2] > 0.97 and values[2] < 0.9996:
                        aos[i].append(j)

    for i in range(len(labels)):
        all_aspect, all_opinion, all_pair, all_apce, pred_aspect, pred_opinion, pred_pair, pred_apce = [], [], [], [], [], [], [], []
        temp_label = labels[i]
        temp_pred = aos[i]
        for j in range(len(temp_label)):
            all_aspect.append([temp_label[j][1]])
            all_opinion.append([temp_label[j][3]])
            all_pair.append([temp_label[j][1], temp_label[j][3]])
            all_apce.append([temp_label[j][1], temp_label[j][2]])
        fall_aspect.append(all_aspect)
        fall_opinion.append(all_opinion)
        fall_pair.append(all_pair)
        fall_apce.append(all_apce)

        for k in range(len(temp_pred)):
            pred_aspect.append([temp_pred[k][1]])
            pred_opinion.append([temp_pred[k][3]])
            pred_pair.append([temp_pred[k][1], temp_pred[k][3]])
            pred_apce.append([temp_pred[k][1], temp_pred[k][2]])
        fpred_aspect.append(pred_aspect)
        fpred_opinion.append(pred_opinion)
        fpred_pair.append(pred_pair)
        fpred_apce.append(pred_apce)

    for i in range(len(aos)):
        labels[i] = list(labels[i])
        aos[i] = list(aos[i])

        result_pre.append(aos[i])
        result_label.append(labels[i])

        n_gold += len(labels[i])
        n_pred += len(aos[i])

        for t in aos[i]:
            if t in labels[i]:
                n_tp += 1
    tp = n_tp
    pred = n_pred
    for m in range(len(aos)):
        for k in range(len(aos)):
            if k >= m:
                for i in range(m, k):
                    n_pred = pred
                    n_tp = tp
                    result_pre[i] = list(result_pre[i])
                    for j in oas[i]:
                        if j in aos[i]:
                            oas[i].remove(j)
                        n_pred += len(oas[i])
                    for j in oas[i]:
                        if j in result_label[i]:
                            n_tp += 1
                    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
                    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
                    f1 = 2 * precision * recall / (
                            precision + recall) if precision != 0 or recall != 0 else 0
                    if f1 > top_f1:
                        top_f1 = f1
                        top_recall = recall
                        top_precision = precision
    print("P:", top_precision, "R:", top_recall, "F1:", top_f1)
    aspect_score = compute_f1_scores(fpred_aspect, fall_aspect)
    print("ATE task P {} R {} F1 {}".format(aspect_score['precision'], aspect_score['recall'], aspect_score['f1']))
    opinion_score = compute_f1_scores(fpred_opinion, fall_opinion)
    print("OTE task P {} R {} F1 {}".format(opinion_score['precision'], opinion_score['recall'], opinion_score['f1']))
    pair_score = compute_f1_scores(fpred_pair, fall_pair)
    print("AOPE task P {} R {} F1 {}".format(pair_score['precision'], pair_score['recall'], pair_score['f1']))
    apce_score = compute_f1_scores(fpred_apce, fall_apce)
    print("APCE task P {} R {} F1 {}".format(apce_score['precision'], apce_score['recall'], apce_score['f1']))

def compute_f1_scores(pred_pt, gold_pt, verbose=True):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])
        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (
        precision + recall) if precision != 0 or recall != 0 else 0
    scores = {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }

    return scores


if __name__ == '__main__':
    args = init_args()
    if args.do_inference:
        fusion(args)

