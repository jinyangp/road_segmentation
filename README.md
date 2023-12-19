<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<div align="center">
  <h2>Road Segmentation - Extract roads from satellite images</h2>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#folder-description">Folder Description</a></li>
        <li><a href="#results">Results</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

For this project, a set of satellite/aerial images were acquired from GoogleMaps. The ground-truth images for each of these images are also provided where each pixel is labelled as road or background. The aim is to train a classifier to segment roads in these images, i.e. assign a label {road=1, background=0} to each pixel.

<p align="center">
  <img src="https://github.com/jinyangp/road_segmentation/assets/85600715/e93e98f3-62de-451b-bc46-eee54345f64d" width="45%" alt="Sample image">
  <img src="https://github.com/jinyangp/road_segmentation/assets/85600715/11f2d618-2b02-4338-894d-2fb9aca2c3fc" width="45%" alt="Sample groundtruth">
</p>
<p align="center">
  <em>Fig 1. Sample image</em> | <em>Fig 2. Sample groundtruth</em>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Folder Description

This section gives a brief description of the content inside each folder.

Due to the lack of local computational resources, notebooks that required the use of an accelerator were executed on Kaggle. Notebooks that were executed on Kaggle are clearly indicated at the top of the notebook.

| Folder name | Description |
| ----------- | ----------- |
| attention_unet | Contains the implementation code for implementing the CBAM-UNet model |
| dataloader | Contains the implementation code to generate datasets and loading them in to be used to train/evaluate the models |
| helpers | Contains the implementation code for helper functions used to generate submissions and visualisations |
| results | Contains the notebooks used for model training, the relevant results and gradient activations |
| u_net | Contains the implementation code for implementing the UNet model |
| generate_submissions.ipynb | Contains the notebook used to generate the submission |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Results

Five models, each labelled as Trial 1-5, were developed iteratively. The table below details the changes made in and the results of each trial.

| Model/Trial no. | Model | Loss Function | Augmented | Patched | Pixel Accuracy | IoU | Dice (F1) |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | 
| 1 | U-Net | Weighted BCE | No | No | 0.925 ± 0.086 | 0.586 ± 0.457 | 0.707 ± 0.227 | 
| 2 | CBAM U-Net | Weighted BCE | No | No | 0.918 ± 0.098 | 0.588 ± 0.511 | 0.699 ± 0.530 |
| 3 | CBAM U-Net | Weighted BCE | Yes | No | 0.921 ± 0.101 | 0.547 ± 0.509 | 0.665 ± 0.522 |
| 4 | CBAM U-Net | Weighted IoU | Yes | No | 0.928 ± 0.092 | 0.762 ± 0.277 | 0.835 ± 0.271 |
| 5 | CBAM U-Net | Weighted IoU | Yes | Yes | 0.863 ± 0.126 | 0.908 ± 0.147 | 0.946 ± 0.110 |

*Note: Metrics (e.g., pixel accuracy) were obtained on a similar validation data set for all trials.* <br>
*BCE - Binary Cross Entropy, IoU - Intersection over Union*

In the end, Model 5 was chosen as the best performing model, with hyperparameters of ```Adam``` optimiser, learning rate of ```0.00001``` and ```100``` epochs.

### Built With

The deep learning models in this project were created using Tensorflow and Keras.

[![Python][Python-img]][Python-url] [![Tensorflow][Tensorflow-img]][Tensorflow-url] [![Keras][Keras-img]][Keras-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

You need to clone this repository.
```sh
git clone https://github.com/jinyangp/dso_localisation.git
```

Due to the lack of computational resources locally, computationally intensive notebooks were ran on Kaggle while the rest were ran locally. 

### Generation of results
To generate the results used in the submission on AICrowd, click on the link made available in ```generate_submissions.ipynb``` and run the notebook. The submission.csv can be found in the Outputs. 

### Notebooks ran on Kaggle
Notebooks ran on Kaggle have been labelled at the top of the notebook. To run Kaggle notebooks, simply click on the link made available at the top of the notebook.

### Notebooks ran locally

First, create a folder ```datasets``` in the base directory with the following structure:

```plaintext
project-root/
│
├── datasets/
│ ├── train/
│ │ ├── groundtruth/
│ │ └── image/
│ ├── test/
│ │ └── image/
```

Each image is labelled as follows 'satImage_xxx.png'.

1. Install virtualenv package
```sh
pip install virtualenv
```

2. Create a virtual environment in desired directory
```sh
cd [project path]
virtualenv venv
```

3. Activate the environment
```sh
source ./venv/bin/activate
```

4. Install Tensorflow
```sh
pip install tensorflow patchify
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

The following past works were referenced:

1. Minaee, S. (2020, January 15). Image segmentation using Deep Learning: A survey. arXiv.org. https://arxiv.org/abs/2001.05566
2. Ronneberger, O. (2015, May 18). U-NET: Convolutional Networks for Biomedical Image Segmentation. arXiv.org. https://arxiv.org/abs/1505.04597
3. Vinogradova, K., Dibrov, A., & Myers, G. (2020). Towards interpretable semantic segmentation via Gradient-Weighted Class Activation Mapping (Student Abstract). Proceedings of the the Thirty-Fourth AAAI Conference on Artificial Intelligence, 34(10), 13943–13944. https://doi.org/10.1609/aaai.v34i10.7244
4. Woo, S. (2018, July 17). CBAM: Convolutional Block Attention Module. arXiv.org. https://arxiv.org/abs/1807.06521

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[Python-img]: https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[Tensorflow-img]: https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[Tensorflow-url]: https://www.tensorflow.org/
[Keras-img]: https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white
[Keras-url]: https://keras.io/
