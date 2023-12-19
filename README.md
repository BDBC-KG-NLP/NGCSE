## Environment

Run the following command to create the required conda environment:

```shell
conda env create -f environment.yml -n your_new_environment_name
```

## Data

+ Train files:

  Run the following command in the `data/` directory:
  
  ```shell
  bash download_hybrid.sh
  ```

+ Evaluation files:
  
  Run the following command in the `SentEval/data/downstream/` directory:
  
  ```shell
  bash download_dataset.sh
  ```

How we hold out 10% of the training data and how some data augmentations are performed are shown in `tools.py`.

## Training

+ Train with final performance in the "Wiki.STS_HT" training setting :

  ```shell
  bash scripts/train_bert_wiki_sts.sh
  ```

+ Train with final performance in the "NLI.STS_HT" training setting :

  ```shell
  bash scripts/train_bert_nli_sts.sh
  ```

The scripts to perform experiments before **Final Performance** section are listed in `scripts/data_domain`.

## Evaluation

```shell
bash scripts/evaluation.sh path_to_the_result
```

## Plotting

How we plot figures in the paper are shown in `plot.py`.