# Environnement de reference

Environnement exact utilise pour les entrainements et les campagnes
finales du rapport. C'est la note de versions demandee par
`docs/DOCKER_DELIVERY.md` : en cas d'ecart de resultats lors d'une
reproduction, comparer d'abord les versions ci-dessous.

## Materiel et OS

- GPU : NVIDIA GeForce RTX 4070 Laptop GPU (8 Go)
- Driver NVIDIA : 591.74
- OS : Windows 11 + PowerShell (venv `aivenv` sur le Bureau, cree par
  `scripts/setup_new_pc_windows.ps1 -UseDesktopAivenv`)

## Logiciels cles

| Composant | Version |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.10.0+cu128 (CUDA 12.8) |
| torchvision | 0.25.0+cu128 |
| Ultralytics | 8.4.14 |
| TensorRT (pip) | 10.16.0.72 |
| onnxruntime-gpu | 1.24.1 |
| onnx | 1.20.1 |
| OpenCV (opencv-python) | 4.13.0.92 |
| NumPy | 2.4.2 |
| SciPy | 1.17.0 |
| matplotlib | 3.10.8 |

La liste complete et exacte des paquets est figee dans
`requirements.lock.txt` (sortie de `pip freeze` de cet environnement).

## Recreer l'environnement

```powershell
python -m venv aivenv
.\aivenv\Scripts\Activate.ps1
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Pour une reproduction stricte, utiliser `requirements.lock.txt` a la
place de `requirements.txt` (necessite le meme index PyTorch cu128 pour
les paquets torch).

## Remarques

- Les engines TensorRT `.engine` livres ont ete generes avec TensorRT
  10.16 sur le GPU ci-dessus : ils ne se chargent pas forcement sur un
  autre GPU ou une autre version de TensorRT. Les regenerer depuis les
  `.pt` avec `Phase_2_Baseline_MonoCam/export_onnx.py` si besoin.
- L'image Docker (`docker/Dockerfile`) installe les memes versions
  epinglees de torch/onnx/onnxruntime/tensorrt sur une base
  `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` ; les wheels cu128
  embarquent leurs bibliotheques CUDA.
- Le re-entrainement n'est pas bit-a-bit deterministe sur GPU : on
  attend des performances comparables, pas des poids identiques.
