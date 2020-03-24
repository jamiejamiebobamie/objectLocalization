# Object Classification and Localization

This is an implementation of Object Classification and Localization using YOLO and MobileNetV2 transfer learning. The final model is flawed, which could be a result of the dataset used to train the model or the chosen architecture, or it could be a result of code (many possiblities: loss function, scaling the coordinates to the aspect ratio of the scaled images, etc.)

## Getting Started

If you would like to try your hand at making it work.

* Clone the repo locally:
```git clone git@github.com:jamiejamiebobamie/objectLocalization.git```
* In your terminal, navigate to the main folder of the cloned repo.
* Install the requirements:
```pip install -r requirements.txt```.
* You'll need the Kitti Dataset. Download it [here](https://www.kaggle.com/twaldo/kitti-object-detection/download) .
* Place the downloaded 'kitti-object-detection' directory in the main project folder.

Your files and folder structure should look like this:
```
main project folder
├── kitti-object-detection          # Downloaded dataset from Kaggle.
│   └── kitti_single               
│       ├── testing
│       │   └── image_2
│       └── training
│           ├── image_2
│           └── label_2
└── ...[ folders and files ]...
```

## Prediction
* To make a prediction type:
```python3 test.py```
in your terminal and press 'enter'.

### Prerequisites

The required packages are listed in the requirements file and are downloaded using the
```pip install -r requirements.txt``` command in your terminal.

## Authors

* **Jamie McCrory**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [lars76](https://github.com/lars76/object-localization) 's implementation of object-localization

![alt text](./results_images/000000.png)
