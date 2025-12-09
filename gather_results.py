import numpy as np
from repetitivness_test import is_likely_llm_by_punctuation

def parse_texts(filepath):
    texts = []
    with open(filepath, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, start=1):
            line = line.rstrip("\n")
            texts.append(line)

    return texts

def parse_file(filepath):
    all_values = []
    with open(filepath, 'r') as f:
        for line in f:
            spl_line = line.strip().split(' ')
            score, cls = spl_line[0], spl_line[1]

            all_values.append((float(score), cls))
    return all_values

def search_threshold(filepaths):
    for fp in filepaths:
        values = parse_file(fp)
        scores = sorted([v[0] for v in values])
        max_acc, best_th = 0.0, -1
        for th in scores:
            # model < th < human
            # print("threshold", th)
            cr = 0
            for (val, label) in values:
                if val <= th and label == 'model':
                    cr += 1
                elif val > th and label == 'human':
                    cr += 1
                else:
                    pass
            
            # print(f'Accuracy: {cr / len(values)}')
            acc = cr / len(values)
            if acc > max_acc:
                max_acc = acc
                best_th = th
        print(max_acc, best_th)
        
def compare_logs():
    filepaths = ['logs/values', 'logs/values2', 'logs/values3']
    thresholds = [0.936, 1.12, 0.936]

    for fp, th in zip(filepaths, thresholds):
        values = parse_file(fp)
        print(fp)
        # model < th < human
        print("threshold", th)
        cr = 0
        for idx, (val, label) in enumerate(values):
            if val <= th and label == 'model':
                cr += 1
            elif val > th and label == 'human':
                cr += 1
            else:
                print(f'Index {idx}, score {val}')
        
        print(f'Accuracy: {cr / len(values)}')

def stats(filepaths, texts):
    th = 0.936
    around_th = 0
    for idx, fp in enumerate(filepaths):
        values = parse_file(fp)
        scores = [score for (score, label) in values]
        
        print(min(scores), max(scores), sum(scores) / len(scores), np.median(scores))
        for idx, (score, label) in enumerate(values):
            if score < th:
                label = "model"
            else:
                label = "human"
                
            if th - 0.02 < score < th + 0.02:
                text = texts[idx]
                is_llm, word_count = is_likely_llm_by_punctuation(text)
                if word_count > 15:
                    new_label = ('model' if is_llm else 'human')
                    print(label, new_label)
                    # print('changed label', idx, score)
                
        # print(sum([1 for sc in scores if th - 0.02 < sc < th + 0.02]))

# compare_logs()
texts = parse_texts('data/test/test_B/data.tsv')
stats(['logs/results_test_b'], texts)


