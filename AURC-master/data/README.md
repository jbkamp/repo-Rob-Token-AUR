Important info about the data:
-	```perturbations_dataset/```	-	-	contains datasets for the robustness tests

Follow the instructions at https://github.com/trtm/AURC to download their dataset and to run the script that generates ```AURC_DATA_dict.json```.
-	```AURC_DATA.tsv```	-	-	origin file by Trautmann used in ```preparation.py``` to generate the original ```AURC_DATA_dict.json```
-	```AURC_DOMAIN_SPLITS.tsv```	-	-	origin file by Trautmann used in ```preparation.py``` to generate the original ```AURC_DATA_dict.json```
-	```AURC_SENTENCE_LEVEL_STANCE.tsv```	-	-	origin file by Trautmann used in ```preparation.py``` to generate the original ```AURC_DATA_dict.json```

Eventually, we work with ```AURC_DATA_dict.json```, which we obtained through the previous steps. We then enrich ```AURC_DATA_dict.json``` (create an updated version of it):
-	```AURC_DATA_dict.json```	-	-	cleaned and enriched dataset (needed to create perturbations datasets) through ```clean_enrich.py```
