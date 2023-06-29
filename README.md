# Uncertainty Aware Deep Learning Classification of Marine Species from ROV Vehicle Videos

[David Serrano](https://scholar.google.es/citations?user=CWuYYNUAAAAJ&hl=en&oi=sra), [David Masip](https://scholar.google.es/citations?user=eHOqwS8AAAAJ&hl=en&oi=ao), José A. Garcia del Arco, Montserrat Demestre, Sara Soto, Andrea Cabrito, Laia Illa-López

[AIWell Research Group, Universitat Oberta de Catalunya](https://aiwell.uoc.edu/)

[Insitut de Ciències del Mar, Spanish National Research Council (CSIC)](https://icm.csic.es/en)

This repositoy contains all the code and software of the paper Uncertainty Aware Deep Learning Classification of Marine Species from ROV Vehicle Videos. The software is designed to classify Marine Species using MonteCarlo Dropout to generate, in addition to the prediction, an associated estimate of uncertainty. 

We experiment with uncertainty estimations and a novel dataset of ROV vehicle images from the Mediterranean Sea, called ICM-20. We find that using uncertainty can make a better use of human annotators efforts when correcting possible misclassified samples.

This repository also contains the code to generate the Correct versus Incorrent Histogram (CIH) and the Accuracy versus Corrected Curve (ACC) proposed in the paper.

## Installation
Create a conda environment named `uncertainty` and install the dependencies:

```bash
```

## Usage
### Configuration files
The entire pipeline uses `.yaml` config files to set the run parameters. These config files are stored in the `config` folder. The config files are divided into five categories: `base`, `model`, `training`, `uncertainty` and `inference`:

- `base` contains the base parameters for the run. These parameters are common to all the runs such as dataset path, class names and wandb parameters.
- `model` contains the model parameters. These parameters are specific to the model architecture.
- `training` contains the training parameters. These parameters are specific to the training process such as learning rate, optimizer, etc.
- `uncertainty` contains the uncertainty parameters. These parameters are specific to the uncertainty estimation process such as the number of samples to use in the MonteCarlo Dropout.
- `inference` contains the inference parameters. These parameters are specific the type of outputs when running the code for inference.

Then a parent config file groups the names of the corresponding files for each category:

```yaml
defaults:
  - base: base_example
  - model: efficientnet_b0_example
  - training: training_example
  - uncertainty: uncertainty_example
  - inference: inference_example
```

Not all config divisions are used in all the runs (train, evaluate, inference). On every folder there is an example file with the explanation of the parameters.

### Log files
The code will create a log folder in the repository in the first run of the code. Then, each run will create a subfolder with the timestamp of run. The log folder will contain all the files generated during the run. There will always be a log.log containing all the information and state of the run. The files according to the type of run are:
```
📂logs/
├── 📂YYYY-MM-DD_HH-MM-SS/ (train.py log)
│   ├── 📜log.log
│   ├── 📜config.yaml
│   ├── 💾epoch_0_validacc_0.5_validf1_0.5.pt
│   ├── 💾 ...
│   ├── 💾epoch_X_validacc_X_validf1_X.pt
│   ├── 🖼️valid_confusion_matrix.jpg
│   ├── 🖼️ACC.jpg
│   ├── 🖼CIH.jpg
│
├── 📂YYYY-MM-DD_HH-MM-SS/ (eval.py log)
│   ├── 📜log.log
│   ├── 📜config.yaml
│   ├── 🖼️test_confusion_matrix.jpg
│   ├── 🖼️ACC.jpg
│   ├── 🖼CIH.jpg
│
├── 📂YYYY-MM-DD_HH-MM-SS/ (inference.py log)
    ├── 📜log.log
    ├── 📜config.yaml
    ├── 🧮video_results.xlsx
    ├── 🧮image_results.xlsx
    ├── 📂class_activation_maps.jpg
        ├── 🖼image1_cam.jpg
        ├── 🖼video1_cam.mp4
```

### Dataset
TODO: Explicar com es descarrega el dataset i com s'ha e configurar...

### Training a model
To train a model, run the following command. This will train a model from scratch if the parameter `model.encoder.params.weights` is not specified. If the parameter is specified, the model will load the weights from the specified path and train from those pretrained weights. The model weights will be saved in the log folder.

```bash
python train.py --config config/config.yaml
```

### Evaluating a model
To evaluate a model, run the following command. This will evaluate the model specified on `model.encoder.params.weights` on the test set and save the results in the log folder.

```bash
python eval.py --config config/config.yaml
```

### Inference
To run inference on a model, run the following command. This will run inference using the model specified on `model.encoder.params.weights` on all the images and videos in the `inference.input_path`. If the parameter `inference.class_activation_maps` is set to any method, the input images or videos will be stored in the log file with the class activation maps heatmaps overlayed. The results will be stored in the log folder. This script also saves an excel file with the predictions and uncertainty estimations for each image and video.

```bash
python inference.py --config config/config.yaml
```

Feel free to modify the configuration files to adapt the parameters based on your specific needs.