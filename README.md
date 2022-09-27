# repo-Rob-Token-AUR
This repo contains the code and data used for the (to be published) paper in the Proceedings of The 9th Workshop on Argument Mining, co-located with COLING 2022 in Gyeongju, South Korea:
```
Kamp, J., Beinborn, L., Fokkens, A. (2022). Perturbations and Subpopulations for Testing Robustness in Token-Based Argument Unit Recognition.
```

Please, consult https://github.com/trtm/AURC for the original AURC-8 dataset where our experiments and diagnostic datasets are based on, and for the respective data preparation scripts (e.g. ```download``` and ```preparation```). Refer to _their_ paper and AURC-8 dataset by citing:
```Trautmann, D., Daxenberger, J., Stab, C., Schütze, H., & Gurevych, I. (2020). Fine-grained argument unit recognition and classification.```

We adapted some of the scripts and added our own scripts. 
```models.py```,```run_AURC_token.py```,```utils.py```, as well as ```run.sh``` are Trautmann et al.'s 2020.

Scripts related to sentence-based modeling (a.k.a. sequence classification) are implemented by us (based on the Hugging Face transformers library), as well as the evaluation scripts, a further cleaning script and sanity checks.
The perturbation datasets are of our own making.
