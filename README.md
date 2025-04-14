# quickdraw-neural-network
simple feedforward neural network trained on google's quickdraw dataset. frontend developed using streamlit on localhost to draw on a canvas and have the network guess your doodle. it's like __% accurate with pretty basic stuff like softmax, relu, and no cnn. project uses no tensorflow or pytorch, just numpy.

quickdraw dataset: https://quickdraw.withgoogle.com/data 
most of the doodles have over a hundred thousand entries, so I use way less (default parameters below)

couple things needed to get this to work:
- numpy (no fancy tensorflow or pytorch)
- sklearn to divide dataset into training and validation
- PIL.Image to help with resizing and grayscale
- quickdraw to import drawing data from google's dataet


general overview: (if you don't know the basics of neural networks this won't make much sense)

# backend
we train the model in quickdraw_dataset.py 

