# Model

## Description

This folder contains the model of the project. The model is a convolutional neural network (CNN) that uses PyTorch as its framework. The model is trained using the [WikiArt dataset provided by Kaggle](https://www.kaggle.com/c/painter-by-numbers) and is able to classify authors by their painting.

## Usage

### Training

To train the model, run the following command:

```bash
python train.py
```

The model will be trained using the default parameters. To change the parameters, use the following command:

```bash
python train.py --batch_size 64 --epochs 10 --lr 0.001
```

### Testing

To test the model, run the following command:

```bash
python test.py
```

The model will be tested using the default parameters. To change the parameters, use the following command:

```bash
python test.py --batch_size 64
```

### Predicting

To predict the class of an image, run the following command:

```bash
python predict.py --image_path path/to/image
```

The model will predict the class of the image using the default parameters. To change the parameters, use the following command:

```bash
python predict.py --image_path path/to/image --top_k 3 --category_names cat_to_name.json
```

## Files

### CNN.pt

This file contains the model architecture. It is used by `train.py` to create the model. The model is created using the `torch.nn` module.

### checkpoint.pt

This file contains the model checkpoint. It is used by `test.py` and `predict.py` to load the model.
