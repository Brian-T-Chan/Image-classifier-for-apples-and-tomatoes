# Image-classifier-for-apples-and-tomatoes
Using a CNN to classify images of apples and tomatoes on limited training data.

In this repository, I used Keras and a CNN to classify images of apples and tomatoes from images of apples and tomatoes collected from Pixabay, Unsplash, Pexels, and Shopify's Burst. The links to all these images can be found in `apple links.txt` and `tomato links.txt`. Generally, I wanted the images to satisfy certain characteristics:

- Although there is a lot of variation in the images, they are mostly kept to a few standard forms: Single fruit at the center, small group of fruits, fruits on a tree, etc. 

- None of the images have both apples and tomatoes. If there are images that do have both fruits, then at least one of the fruits is/are not recognizable to the human eye.

I ran several tests on certain collections of images (datasets) from those links. Generally, I considered balanced datasets (same number of apples as tomatoes) with the number of images of each fruit ranging from around 125 images to 210 images. I would first use `preprocessing.py` to compress these images to 150 by 150 pixel images, resulting in datasets that were at most around 6 MB on disk. The file `preprocessing.py` uses `compress.py`, which uses pandas groupby operations to compress the images.

I would then take 10% of such compressed images as test data, and with the remaining compressed data, perform an 80-20 split so that 80% of the remaining data was training data and 20% of the remaining data was validation data. I made this 80-20 split because I wanted to use an early stopping callback during training, which relies on validation data. 

Throughout, I used a hinge loss function, used 200 epochs of training, and set the patience for the early stopping callback to 100 epochs. This is detailed in `CNN.py`, where I use a CNN to determine whether the images from the test data were apples or tomatoes.

The accuracies of such tests for one such dataset are listed below (this dataset had 210 apple images and 210 tomato images). For every test run, different subsets of the dataset were selected to be the training data, validation data, and test data, accounting for the varying test accuracies.

`Test accuracy: 0.7857142686843872`<br>
`Test accuracy: 0.7142857313156128`<br>
`Test accuracy: 0.738095223903656`<br>
`Test accuracy: 0.7142857313156128`<br>
`Test accuracy: 0.738095223903656`<br>
`Test accuracy: 0.8571428656578064`<br>
`Test accuracy: 0.738095223903656`
