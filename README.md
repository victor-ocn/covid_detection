# About our project

We used the ['COVID-19 Radiography Database'](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle wich is a database of chest X-ray images for COVID-19 positive cases along with Normal, Viral Pneumonia and Lung Opacity images.

This dataset is composed by:
- 3,616 COVID-19 images;
- 10,192 Normal images;
- 6,012 Lung Opacity images;
- 1,345 Viral Pneumonia images.

In order to increase this data, we take some other x-ray images for COVID and Pneumonia.

- [Link 1](https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png)
- [Link 2](https://github.com/armiro/COVID-CXNet/tree/master/chest_xray_images/covid19)
- [Link 3](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
)
 - [Link 4](https://www.kaggle.com/code/ibrahimsobh/chest-x-ray-covid19-efnet-densenet-vgg-grad-cam/input)

 To further enhance the dataset, additional COVID-19 and Pneumonia images were incorporated, resulting in a final dataset of over 25,000 images composed by:

- 4,319 COVID-19 images;
- 10,192 Normal images;
- 6,012 Lung Opacity images;
- 4,763 Viral Pneumonia images.

The COVID-19 detection app utilizes a Convolutional Neural Network (CNN) model to identify patterns in X-ray images.

```
model = Sequential()
    model.add(Rescaling(1./255, input_shape=input_shape))

    model.add(layers.Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, kernel_size=(3,3), padding='same', activation="relu"))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, kernel_size=(3,3), padding='same', activation="relu"))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(4, activation='softmax'))
```
This robust dataset, combined with advanced deep learning techniques, enables the app to deliver reliable and accurate results in the detection of COVID-19.

You can try DetectionApp [here](https://coviddetection-a8h898awmch9rtpddcrkpc.streamlit.app/)
