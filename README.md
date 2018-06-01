# Musical-Sub-Genre-Classificator-NET

A simple Convolutional Neural Network for Metal Sub-genre classification based on artwork (Covers).

This is a simple implementation of a CNN with python / Keras. This repo does not include the data. Feel free to contact me if you want it


## Prerequisite

* python3
* [keras](https://keras.io/) (not the version 1.0)
* [skimage](http://scikit-image.org/)
* [setuptools](https://pypi.org/project/setuptools/)

```
pip3 install setuptools
pip3 install keras
pip3 install scikit-image
```

## Install the predictor


```
git clone https://github.com/ZarebskiDavid/Musical-Sub-Genre-Classificator-NET.git
cd Musical-Sub-Genre-Classificator-NET
sudo python3 setup.py install
```

This will create a *build*, a *dist* and a *stylecheckerm.egg-info* directory in the MapreducePyConsole folder.

## Usage

You can call stylecheckerm directly from your terminal

```python
stylecheckerm [path_to_image]
 ---- [path_to_image]: path to a bitmap (png, jpg, jpeg) type of image (e.g. ~/Images/covers/cover.png)
```

## Uninstall
```
sudo rm -R Musical-Sub-Genre-Classificator-NET
```
