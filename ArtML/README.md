# ArtML

ArtML is a Machine Learning project that uses a Neural Network to identify the author of painting. The project is currently in development and is not yet ready for use.

## Requirements specification

*Requirements below are numbered according to their priority. The priority is indicated by the number in square brackets, with 1 being the highest priority and 3 being the lowest.*

### Functional requirements

- [ ] The system should be able to classify the author of a painting based on the dataset. [1]
- [ ] The system should be able to take an input image of a painting and output the predicted artist who created it. [2]
- [ ] The system should be able to handle different types of paintings, such as oil paintings, watercolors, or sketches. [3]
- [ ] The system should be able to handle paintings from different time periods and styles. [3]

### Non-functional requirements

- [ ] The system should be reliable and consistently produce accurate predictions. [2]
- [ ] The system should have a fast response time, with predictions made in a reasonable amount of time. [1]
- [ ] The system should be able to handle large datasets efficiently, without requiring excessive computational resources. [3]
- [ ] The system should be user-friendly, with a simple and intuitive interface for users to input their images and view the predicted artist. [3]

## Architecture

### Data

The data used for this project most likely will be the [Painter by Numbers](https://www.kaggle.com/c/painter-by-numbers) dataset from Kaggle. The dataset contains 103250 paintings from 3000 artists. The dataset is split into a training set and a test set. The training set contains 79433 paintings and the test set contains 23817 paintings.

[//]: # (### Model)

[//]: # (The model used for this project is a Convolutional Neural Network. The model is trained on the training set and validated on the test set. The model is trained using the [Adam]https://arxiv.org/abs/1412.6980 optimizer and the [Categorical Crossentropy]https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression loss function. The model is trained for 100 epochs with a batch size of 32.)

[//]: # (### User interface)

[//]: # (The user interface is a web application that allows the user to upload a painting and get the author of the painting as well as the confidence level of the prediction.)

[//]: # (## Development)

[//]: # (### Prerequisites)

[//]: # (- [Python 3.8]https://www.python.org/downloads/release/python-380/)
[//]: # (- [Pipenv]https://pipenv.pypa.io/en/latest/)
[//]: # (- [Node.js 14.15.4]https://nodejs.org/en/download/)
[//]: # (- [Yarn 1.22.10]https://classic.yarnpkg.com/en/docs/install/#windows-stable)
[//]: # (- [Git]https://git-scm.com/downloads)
[//]: # (- [Visual Studio Code]https://code.visualstudio.com/download)

[//]: # (### Setup)

[//]: # (1. Install [Python 3.8]https://www.python.org/downloads/release/python-380)
[//]: # (2. Install [Pipenv]https://pipenv.pypa.io/en/latest/)
[//]: # (3. Install [Node.js 14.15.4]https://nodejs.org/en/download/)
[//]: # (4. Install [Yarn 1.22.10]https://classic.yarnpkg.com/en/docs/install/#windows-stable)
[//]: # (5. Install [Git]https://git-scm.com/downloads)
[//]: # (6. Install [Visual Studio Code]https://code.visualstudio.com/download)
[//]: # (7. Clone the repository)
[//]: # (8. Open the repository in Visual Studio Code)
[//]: # (9. Open a terminal in Visual Studio Code)
[//]: # (10. Run `pipenv install`)
[//]: # (11. Run `yarn install`)
[//]: # (12. Run `yarn build`)
[//]: # (13. Run `pipenv run python manage.py migrate`)
[//]: # (14. Run `pipenv run python manage.py runserver`)
[//]: # (15. Open a browser and go to <http://>)
[//]: # (16. Upload a painting)
[//]: # (17. Wait for the prediction)
[//]: # (18. The author of the painting and the confidence level of the prediction should be displayed)
[//]: # (19. If the prediction is not displayed, run `pipenv run python manage.py runserver` again)

## License

*MIT Licence(X11 Licence)*

## Authors

- [**Felix Manning**](https://github.com/Felix-1871)
- [**Adam Chocholski**](https://github.com/AdamChocholski)
- [**Mateusz Maleszewski**](https://github.com/Mateusz022)

## Acknowledgments

- [**Kaggle**] for providing the dataset
- [**PyTorch**] for providing the machine learning framework
- [**Django**] for providing the web framework
-

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
