# Donnees et securite

## Ce qui est volontairement exclu

- Videos RTSP et videos de test.
- Images et labels des datasets complets.
- Poids de modeles `.pt`, `.pth`.
- Engines TensorRT `.engine`.
- Exports frame-par-frame volumineux.
- Logs complets contenant des details d'execution.

Ces fichiers doivent etre stockes hors Git, par exemple dans un volume Docker, un disque partage ou un stockage objet.

## Donnees conservees dans Git

- Code source des phases.
- Ground truths JSON legeres.
- Configurations de calibration assainies.
- CSV synthetiques de resultats.
- Documentation de methode.

## Assainissement effectue

Les valeurs suivantes ont ete remplacees par des placeholders :

- identifiants RTSP ;
- mots de passe camera ;
- IPs internes ;
- URLs camera completes.

Les placeholders utilises sont :

```text
<USER>
<PASSWORD>
<CAMERA_IP>
```

Avant d'executer les scripts sur une installation reelle, creer une configuration locale non versionnee, par exemple `config.local.yaml`, et y renseigner les vraies URLs RTSP.

