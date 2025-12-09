from mlx_lm import load, generate

import nltk

from binoculars import Binoculars
from model import HFModel

N_samples = 1000
texts, responses = [], []
data_path, labels_path = 'data/train/data.tsv', 'data/train/labels.tsv'

with open(labels_path, "r", encoding="utf-8") as file:
    for i, line in enumerate(file, start=1):
        line = line.rstrip("\n")
        responses.append(line)

        if i > N_samples:
            break


with open(data_path, "r", encoding="utf-8") as file:
    for i, line in enumerate(file, start=1):
        line = line.rstrip("\n")
        texts.append(line)

        if i > N_samples:
            break


# model_name_1 = "speakleash/Bielik-4.5B-v3"
# model_name_2 = "speakleash/Bielik-4.5B-v3.0-Instruct"
model_name_1 = "speakleash/Bielik-11B-v2"
model_name_2 = "speakleash/Bielik-11B-v2.3-Instruct"

bino = Binoculars(model_name_1, model_name_2, use_bfloat16=False, mode='accuracy')

th = 0.936
correct_responses, crt_rsp_2 = 0, 0
for text, resp in zip(texts, responses):
    pred, score, token_counts = bino.predict(text)
    pred = pred.strip()

    pred = (0 if pred == 'human' else 1)
    ground = (0 if resp == '0' else 1)

    print(f'Predicted: {pred}')
    print(f'Ground: {ground}')

    score = score.tolist()
    score = round(score, 3)
    print(f'{score} {ground} {token_counts}')

    if pred == ground:
        correct_responses += 1

print("ACC", correct_responses / len(responses))
