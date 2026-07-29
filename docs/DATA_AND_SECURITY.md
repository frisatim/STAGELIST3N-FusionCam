# Donnees et securite

## Ce qui est volontairement exclu

- Videos RTSP et videos de test.
- Images et labels des datasets complets.
- Poids de modeles `.pt`, `.pth`.
- Engines TensorRT `.engine`.
- Exports frame-par-frame volumineux.
- Logs complets contenant des details d'execution.

Ces fichiers sont livres hors Git dans le dossier externe
`STAGELIST3N-FusionCam-data` (voir `docs/DATA_LAYOUT.md`), par exemple
sur un disque partage, une cle USB ou un drive.

## Donnees conservees dans Git

- Code source des phases.
- Ground truths JSON legeres.
- Configurations de calibration assainies.
- CSV synthetiques de resultats et figures.
- Documentation de methode.

## Assainissement effectue

Les valeurs suivantes ont ete remplacees par des placeholders, dans le
code, les configurations et la documentation (y compris les notes de
travail de `docs/archive/`) :

- identifiants RTSP ;
- mots de passe camera ;
- IPs internes du laboratoire ;
- URLs camera completes ;
- noms d'infrastructure interne.

Les placeholders utilises sont :

```text
<USER>
<PASSWORD>
<CAMERA_IP>
<CAMERA_NET>
<GATEWAY_IP>
<SERVER_IP>
```

Avant d'executer les scripts sur une installation reelle, creer une
configuration locale non versionnee, par exemple `config.local.yaml`,
et y renseigner les vraies URLs RTSP.

## Verification avant publication

La commande de controle est donnee dans
`docs/FINAL_DELIVERY_CHECKLIST.md` (grep des motifs sensibles + liste
des fichiers lourds). Elle doit ressortir vide avant tout snapshot
public du depot.
