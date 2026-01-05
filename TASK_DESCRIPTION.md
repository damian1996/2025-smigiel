
# ŚMIGIEL: Spotting Machine-Generated Text from LLMs for Polish

Organising team: 🦸🏻‍♂️[Piotr Przybyła](https://piotr.phd/) 👨🏼‍💻[Jakub Strebeyko](https://github.com/JStrebeyko)
👩🏻‍🏫[Alina Wróblewska](https://zil.ipipan.waw.pl/AlinaWroblewska)

## 👋 Introduction

**\*śmigiel** ['ɕmiɡjɛl] m I, D. ~gla gw. «płaski szczebel w drabinach wozu» (Eng. flat rung in a ladder wagon)
// SJP W. Doroszewskiego\*

The rapid progress of large language models (LLMs) in recent years has enabled the generation of highly fluent and linguistically correct texts in numerous languages. Although these models demonstrate strong performance in natural language processing (NLP) and natural language generation (NLG) tasks, their reliance on human-authored data and their capacity to emulate human writing styles raise critical concerns regarding authenticity, authorship attribution, and the potential for misuse. In response to these challenges, there is a growing research focus on developing systems for machine-generated text detection — tools designed to distinguish between human-authored and AI-generated content.

With the ongoing development of machine-generated text (MGT) systems, the need for their evaluation and benchmarking has become increasingly important. This challenge has already given rise to several shared tasks, e.g. [GenAI Content Detection](https://genai-content-detection.gitlab.io) at COLING 2025, [SemEval-2024 Task 8](https://github.com/mbzuai-nlp/SemEval2024-task8), and [PAN 2025 Task 1](https://pan.webis.de/clef25/pan25-web/style-change-detection.html). However, the majority of these initiatives have focused primarily on English texts. An important exception and a source of inspiration for our task is [IberAuTexTification](https://sites.google.com/view/iberautextification/home), which targets the languages of the Iberian Peninsula. The question of how difficult MGT detection is for the Polish language remains open.

## 💡 Task description

Here, we outline ŚMIGIEL: a shared task on **S**potting **M**achine-**G**enerated Text from **L**LMs for Polish, organised at the [PolEval 2025](https://poleval.pl) evaluation campaign.

### Objective

The main objective of this shared task is to benchmark and enhance the state-of-the-art in detecting machine-generated texts in the Polish language across various domains and textual genres. Robust and reliable MGT detection systems will undoubtedly contribute to the broader goals of responsible AI development, supporting critical areas such as media verification, academic and journalistic integrity, and, potentially, digital forensics.

### Procedure

The task is framed as a binary classification problem: distinguishing between human-authored and machine-generated texts. Participating systems are given a collection of text fragments, and must assign each either label 0 (human-written) or label 1 (LLM-generated). To foster the development of models capable of generalising across diverse writing styles, the test dataset includes samples from domains not represented in the training set.

### ŚMIGIEL subtasks

The participants can choose to submit their solutions in three subtasks:

1. **UNSUPERVISED**: classifiers that are prepared without the use of training data,
2. **CONSTRAINED**: classifiers that are trained only on the dataset provided by the organisers,
3. **OPEN**: classifiers that are trained in any way, including external datasets and data augmentation.

Submissions within these subtasks will be evaluated and ranked separately.

### Task constraints

1. The use of **publicly available** pretrained models, both Polish-specific and multilingual, is allowed.
1. Participants may use **publicly accessible** Polish corpora, lexical resources, knowledge bases, and other structured data resources.
1. Participants are expected to prepare a short article, describing their solution with enough details to allow replication of the research.
1. All external models and resources used must be listed in the submission, including bibliographic references or direct links.
1. The use of proprietary or non-public datasets, models, or services is strictly prohibited.
1. Each team is allowed a maximum of three submissions per subtask.

## 📦 Dataset

### Content

Human-written texts originate from various sources and are available under open licenses for research purposes. The texts cover the following domains:

- customer reviews
- literature
- social media
- wikipedia

Machine-generated texts are prepared using a range of open-source LLMs of different sizes:

- small: [LLama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [Bielik 7B](https://huggingface.co/speakleash/Bielik-7B-Instruct-v0.1), [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- medium: [Bielik 11B](https://huggingface.co/speakleash/Bielik-11B-v2.3-Instruct), [Mistral Nemo](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407), [PLLuM 12B](https://huggingface.co/CYFRAGOVPL/pllum-12b-nc-chat-250715)
- large: [Gemma 3 27B](https://huggingface.co/google/gemma-3-27b-it).

The dataset is balanced as follows:

- It contains an equal proportion of human-written and LLM-generated instances.
- Domains are uniformly represented.
- It includes roughly one-third of the LLM-generated texts from each model size.

### Format

Each text to be classified is placed on a separate line in the ``data.tsv`` file, while corresponding labels are stored in the ``labels.tsv`` file.

All texts, whether LLM-generated or human-written, are proportionally shortened to ensure consistency and to minimise authorship clues. This is achieved by truncating the content once it reaches a predefined character limit.

**Training examples**

| #   | text                                                                                                                                                                                                                                                      | label |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| 1   | *The book discusses the statistical revolution which took place in the twentieth century, where science shifted from a deterministic view (Clockwork universe) to a perspective concerned primarily with probabilities and distributions and parameters.* | 0  |
| 2   | *I don’t eat, sleep, or forget things mid-sentence (unless you want me to role-play as someone who does).*                                                                                                                                                | 1   |

### Test data

Test data will be similar in nature, but will also be based on models and genres unseen during training.

**Test examples**

| #   | text                                                   |
| --- | ------------------------------------------------------ |
| 1   | *If I can fool you, does it matter if I am a machine?* |
| 2   | *The war of Gog and Magog has been foreseen aeons ago* |

## ⚖️ Evaluation

### Metrics

In the ŚMIGIEL task, ``Accuracy`` is used as the primary evaluation metric. Accuracy, defined as the proportion of correctly classified instances over the total number of instances, is widely used in text classification research (Jurafsky and Martin, 2023). 

### Baseline

The organisers will prepare some simple baseline solutions, relying on mainstream approaches to text classification.


## 🚀 Submission

### Test data

* [test A](https://github.com/poleval/2025-smigiel/tree/main/data/test/test_A)
* test B

📢 <span style="color:red">**Update 2025-11-04**</span>: Participants who wish to obtain access to the test B data and evaluate their final solutions are kindly asked to complete a [short survey](https://docs.google.com/forms/d/e/1FAIpQLSfzO4jCb3AtFAcUvQdFbATR0HRcD9uN9vEbtfa0uUPBM6h-fw/viewform) by November 6, 2025.

### Submission format − example

* [Sample submission file](https://github.com/poleval/2025-smigiel/blob/main/extras/out.tsv)

### PolEval 2025 evaluation platform

* [Śmigiel − Test A](https://poleval.amueval.pl/challenge/Spotting%20Machine-Generated%20Text%20from%20LLMs%20–%20Test%20A)
* Śmigiel − Test B

## 📚 Literature

**Jurafsky, D. & Martin, J. H. (2024).** [_Speech and Language Processing. An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models_](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf). 3rd edition.

**Moreno-Sandoval, A., García-Vega, M., Casado-Molina, A., García-Hernández, R., García-Rodríguez, R., & García-Pablos, A. (2023).** [_Overview of AuTexTification at IberLEF 2023: Detection and attribution of machine-generated text in multiple domains._](https://arxiv.org/abs/2309.11285) arXiv.

**Sarvazyan, A. M., González, J. Á., Franco-Salvador, M., Rangel, F., Chulvi, B., & Rosso, P. (2023).** [_Overview of AuTexTification at IberLEF 2023: Detection and attribution of machine-generated text in multiple domains._](arXiv. https://arxiv.org/abs/2309.11285) arXiv.

**Wang, Y., Mansurov, J., Ivanov, P., Su, J., Shelmanov, A., Tsvigun, A., Mohammed Afzal, O., Mahmoud, T., Puccetti, G., & Arnold, T. (2024).** _[SemEval-2024 Task 8: Multidomain, Multimodal and Multilingual Machine-Generated Text Detection.](https://doi.org/10.18653/v1/2024.semeval-1.279) In Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024)_ (pp. 2057–2079). Association for Computational Linguistics.