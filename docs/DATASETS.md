# Datasets

Les datasets complets ne sont pas inclus dans ce depot Git.

Structure locale recommandee :

```text
recordings/recordings/              Videos camera brutes.
dataset/                            Dataset personne / zones.
dataset_objets_HD/                  Dataset objets interdits.
dataset_output_batch/               Extractions images/labels par camera.
Phase_2_Baseline_MonoCam/Modelstrained/
                                   Poids entraines et engines exportes.
```

Les fichiers de ground truth legers sont places dans `ground_truth/`.

Pour reproduire une campagne, il faut replacer les videos et poids aux chemins attendus ou adapter les arguments CLI des scripts.

