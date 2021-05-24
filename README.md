# Prediction of Wildfire Evolution with a Cellular Automaton-based Model Learning from Satellite Data

This repository contains most of the code used in our project that tried to predict the evolution of wilfires with a statistical model that learns from satellite data. The README below details how each of the files contribute to the project. You will also find some figures showing what the data and model look like.

What you will find in this repository :

# DataCollectionAndProcessing

`DataCollectionAndProcessing` contains scripts related to the collection and processing of fire data.

## firedata_automation.py

firedata_automation.py implements the functions needed to extract and process fire data from Copernicus.

### The class `Fire_spot` 

The `Fire_spot` class implements the part that downloads the images before and after the fire. Its main fields are :
- `lat`: the latitude of the burned spot
- `lng`: the longitude of the burned spot 
- `date`: the date of the fire
- `radius`: the radius of the area from which we want to obtain its data
- `deltatime`: the length of time in which we collect data for both period befor and after the fire event 

The method `get_product` retrieves the image with the least cloud cover between the available
products corresponding to the selected spot and period.


### The class `Image_processing` 

`Image_processing` implements the functions that calculates vegetation density and other important indeces. Its constructor gets an instance of `Fire_spot`
class. Its main method is:

- `calc_indices`: it calculates all the indices we need and store theme in the dictionary field `bands_bfr` or `bands_aftr`.

The static method `run` in the bottom of that file runs the whole algorithm on the FIRMS data.


# DeterministicModel

`DeterministicModel` contains the implementation of the model described in 5.4

# LearningTest

`LearningTest` contains the implementation of the first model described from 5.1 to 5.3 and in 5.5.

## automata.py

Automata.py implements the basic data structures needed to work with the inital model, namely `Cell` and `Automaton`.

### The class `Cell`

`Cell` is, as explained in the report, the description of a cell of the automata. It simply contains all the information in that cell:
- `moisture` : the value of moisture for that cell
- `fuel` : the remaining fuel in that cell
- `state` : the current state of the cell

The state of the cell is coded by a integer:
- 1 : flammable
- 2 : burning
- 3 : burnt


### The class `Automaton`

`Automaton` is the class that implements the cellular automaton. 

Its most important fields are:

- `mat`: The matrix of `Cell` that represents the current state of the automaton.
- `c_intercept`: The constant coefficient.
- `c_moisture`: The coefficient associated to moisture.
- `neighborsMatrix`: The matrix that stores the indices neighbors of each index.
- `cache`: The `set` that contains the indices of the cells that need to be updated during the next time step.

Its most important methods are :

- `time_step`: Simulates a time step following the rules described in the report (5.1 and 5.2).
- `final_step`: Simulates the automaton until there is no burning cells left and returns the matrix of the state of the cells (1 for not burnt and 3 for burnt).

The file also contains code to represent the evolution of the fire in a graphic window.


## data_generator.py

data_generator.py provides a function to generate artificial and credible data in order to experiment with learning.

The most important functions are:

- `perlin`: Returns a perlin noise matrix with the specified seed
- `data`: Generates `n` entries of data and returns a `pandas.DataFrame` containing them. Each entry is created by generating of map of moisture with `perlin` and computing the associated target value (that is, the map of burnt cells) by simulating an automaton with the map of moisture and the parameters `c_intercept` and `c_moisture` specified.


## machine.py

The class `machine` implements the gradient descent to learn the correct parameter from the data generated with the `data` function of `data_generator.py`.

Its fields are:

- `data`: The data to optimize the parameters over.
- `c_intercept` and `c_moisture`: The parameters to optimize.
- `h`: The small number in the first order approximation of the derivative.
- `learning_rate`: The learning rate for the gradient descent.
- `mbsize`: The size of the mini-batch selected at each gradient descent step to compute the gradient over.

If `remote` is True but not `cluster`, the program will be distributed locally. If both `remote` and `cluster` are set to True, the code will be run on the ray cluster (provided it has been initialized with `ray up config.yaml`).


Its most important methods are:

- `fullcost`: Returns the average of the cost of each data point, each cost being itself averaged over `m` simulations. Depending on the state of `remote`, fullcost either calls a method that does the computation normally or one that does it remotely.
- `learn_step`: Performs one step of the gradient descent, by first computing the gradient over a selected mini-batche (the gradient on each instance in the mini-batch being averaged over `m` simulations), and then using the gradient to take a step in the parameter space in (hopefully) the right direction.

Both these method call on different submethods depending on whether the calculations are done remotely or not. In the former case, it eventually calls functions instead of method because it is easier to work with functions when doing remote computations.

The `matdiff` function returns the square of the number of cells in which the state (burnt or not burnt) is different. It is used as the error function to quantify how far the prediction strays from the target value.

The "main function" builds a database with the `data` function from `data_generator.py` and a `machine` object. It then performs learning steps and displays information about the evolution of the parameters and the cost.


# newenv

newenv is an environment that can be used to run the python code of LearningTest and TrueMachine.py. The only library it does not contain is the one that enable us to display graphics on a remote computer (I forgot its name) because it was too big for github.

# processing_result

It contains the processed fire data needed for learning. 
The Australia_fire_xxx files contain 3 .csv files that provide information on the fire and a .png that gives a visual representation of the data.
processed_wind.txt contains information on the wind values for each entry of Australia_fire_xxx.

# config.yaml

config.yaml is the configuration file needed to start the ray cluster with the command `ray up config.yaml`. To make it work on your own account, make sure to change the paths and to provide a path to your connection key.

# Simulation result

Here are the result of the simulation of the cellular automata described above on some of the data in processed_result:

![ani8](https://user-images.githubusercontent.com/73946504/117882016-30edb000-b2aa-11eb-8fc6-7a6946075b97.gif)

![ani9](https://user-images.githubusercontent.com/73946504/117882008-2fbc8300-b2aa-11eb-8765-d2f6be0dbd30.gif)

![ani10](https://user-images.githubusercontent.com/73946504/117882229-68f4f300-b2aa-11eb-908b-b3c2e7437c66.gif)

![ani13](https://user-images.githubusercontent.com/73946504/117882244-6d211080-b2aa-11eb-9101-bf72a5d5e0d4.gif)

![ani14](https://user-images.githubusercontent.com/73946504/117882261-70b49780-b2aa-11eb-8724-faefa7fb9bbe.gif)


And some less interesting ones for the purpose of representativity:

![ani0](https://user-images.githubusercontent.com/73946504/117882355-89bd4880-b2aa-11eb-9ca0-8466bf475f87.gif)

![ani1](https://user-images.githubusercontent.com/73946504/117882356-8a55df00-b2aa-11eb-93b8-605431d15bc2.gif)

### An example of different end results for simulations with identical parameters and data

![Ani](https://user-images.githubusercontent.com/73946504/119347397-c4a48080-bc9b-11eb-8a26-24c52bd9ee76.gif)

![Ani2](https://user-images.githubusercontent.com/73946504/119347409-c9693480-bc9b-11eb-9b6d-fc6368eb98bc.gif)

