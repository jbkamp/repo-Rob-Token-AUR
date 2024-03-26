# repo-Rob-Token-AUR
This repo contains the code and data used for the (to be published) paper in the Proceedings of The 9th Workshop on Argument Mining, co-located with COLING 2022 in Gyeongju, South Korea:
```bibtex
@inproceedings{kamp-etal-2022-perturbations,
    title = "Perturbations and Subpopulations for Testing Robustness in Token-Based Argument Unit Recognition",
    author = "Kamp, Jonathan  and
      Beinborn, Lisa  and
      Fokkens, Antske",
    editor = "Lapesa, Gabriella  and
      Schneider, Jodi  and
      Jo, Yohan  and
      Saha, Sougata",
    booktitle = "Proceedings of the 9th Workshop on Argument Mining",
    month = oct,
    year = "2022",
    address = "Online and in Gyeongju, Republic of Korea",
    publisher = "International Conference on Computational Linguistics",
    url = "https://aclanthology.org/2022.argmining-1.5",
    pages = "62--73",
    abstract = "Argument Unit Recognition and Classification aims at identifying argument units from text and classifying them as pro or against. One of the design choices that need to be made when developing systems for this task is what the unit of classification should be: segments of tokens or full sentences. Previous research suggests that fine-tuning language models on the token-level yields more robust results for classifying sentences compared to training on sentences directly. We reproduce the study that originally made this claim and further investigate what exactly token-based systems learned better compared to sentence-based ones. We develop systematic tests for analysing the behavioural differences between the token-based and the sentence-based system. Our results show that token-based models are generally more robust than sentence-based models both on manually perturbed examples and on specific subpopulations of the data.",
}
```

Please, consult https://github.com/trtm/AURC for the original AURC-8 dataset where our experiments and diagnostic datasets are based on, and for the respective data preparation scripts (e.g. ```download``` and ```preparation```). Refer to _their_ paper and AURC-8 dataset by citing:
```Trautmann, D., Daxenberger, J., Stab, C., Schütze, H., & Gurevych, I. (2020). Fine-grained argument unit recognition and classification.```

We adapted some of the scripts and added our own scripts. 
```models.py```,```run_AURC_token.py```,```utils.py```, as well as ```run.sh``` are Trautmann et al.'s 2020.

Scripts related to sentence-based modeling (a.k.a. sequence classification) are implemented by us (based on the Hugging Face transformers library), as well as the evaluation scripts, a further cleaning script and sanity checks.
The perturbation datasets are of our own making.
