# quickdraw-neural-network
simple feedforward neural network trained on google's quickdraw dataset. frontend developed using streamlit on localhost to draw on a canvas and have the network guess your doodle. it's like __% accurate with pretty basic stuff like softmax, relu, and no cnn. project uses no tensorflow or pytorch, just numpy.

quickdraw dataset: https://quickdraw.withgoogle.com/data 
most of the doodles have over a hundred thousand entries, so I use way less (default parameters below)

notable imports:
- numpy (no fancy tensorflow or pytorch)
- sklearn to divide dataset into training and validation
- PIL.Image to help with resizing and grayscale
- quickdraw to import drawing data from google's dataset
- streamlit for canvas and interactivity


general overview: (if you don't know the basics of neural networks this won't make much sense)

# backend
1. fetch data using quickdrawdatagroup
    - if folders already exist in the same directory as 'quickdraw_dataset.py' with the hierarchy quickdraw/data for each doodle type (such as 'cat' or 'car) there will be no additional files donwloaded
2. convert data from vector into 28x28 png and a stroke width of 3 which helps increase variation for training
3. create neural network using feed forward structure where each neuron connects to every other neuron
    - input layer with 28x28 pixels (784) inputs
    - each of the 3 hidden layers utilizes ReLU activation
    - use softmax at the end to turn raw score into probabilities
4. calculate prediction accuracy with cross entropy loss
5. backpropagation with one hot encoding
    - update weights and biases accordingly
6. save model to model.npz
7. training loop
    - create a batch
    - forward pass
    - compute loss
    - backpropagation
    - update weights
    - track accuracy and print for each epoch

# frontend (streamlit)
1. import 'load_model" from 'quickdraw_dataset.py' as well as streamlit, PIL.Image, and streamlit_drawable_canvas
2. create 700x700 white canvas with black stroke color
3. process canvas input
    - convert canvas to grayscale, resize to 28x28 (model input), and flatten array
    - load model and feed array
4. select highest probablity and print result

# default parameters and utilization
1. first things first you need to use pip to install streamlit, numpy, Pillow,streamlit-drawable-canvas, and scikit-learn
2. go to https://quickdraw.withgoogle.com/data and figure out what doodles you want to train the model on, these will be our classes
3. go to line 172 in 'quickdraw_dataset.py' and change classes. use the same order when defining classes in line 17 of 'quickdraw_streamlit.py'
4. make a folder data/quickdraw in the directory where 'quickdraw_dataset.py' and 'quickdraw_streamlit.py' are located
5. run 'quickdraw_dataset.py' from terminal to begin training
6. create a terminal at the parent folder of all files and run "streamlit run quickdraw_streamlit.py"

