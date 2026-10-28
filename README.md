# CIFAR-10 Image Classification with CNNs

## Interactive Streamlit app

A companion application is available in [streamlit_app/app.py](streamlit_app/app.py).

```bash
python -m pip install -r streamlit_app/requirements.txt
python -m streamlit run streamlit_app/app.py
```

Image inference requires a compatible trained .keras checkpoint. A compact CIFAR-10 training/export script is included; no pretrained checkpoint is committed.

See [DEPLOYMENT.md](DEPLOYMENT.md) for Community Cloud setup, validation coverage, and runtime requirements. A live deployment is not yet confirmed.


A TensorFlow/Keras notebook exploring image classification on CIFAR-10 using convolutional layers.

## Why this project matters

Image classification connects data preparation, neural network design, training, and class-level evaluation in one practical workflow.

## Technical focus

Python · TensorFlow/Keras · NumPy · Matplotlib · classification reports

## Explore the implementation

Open [the notebook](<CNN.ipynb>) to inspect preprocessing, implementation, and saved outputs.

## Run locally

```bash
git clone https://github.com/Arjun1443/CNN.git
cd CNN
python -m venv .venv
# Activate .venv before continuing.
python -m pip install tensorflow numpy matplotlib scikit-learn jupyter
python -m jupyter notebook
```

Select `CNN.ipynb` in Jupyter and review its configuration before executing cells in order.

## Reproducibility notes

CIFAR-10 is loaded through `tensorflow.keras.datasets.cifar10.load_data()` and requires a dataset download on the first run. Review training settings and compare per-class precision and recall rather than relying only on accuracy.

Saved notebook outputs describe a prior execution; they are not an independently reproduced benchmark. Dependency versions are not locked, so compatibility may require adjustment.


## Author

[Naga Anweesh Reddy](https://github.com/Arjun1443/Naga-Anweesh-Reddy) · AI development and technical training
