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

- `calc_indices`: it calculates all the indeces we need and store theme in the dictionary field `bands_bfr` or `bands_aftr`.

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
- `learn_step`: 


# newenv

newenv is an environment that can be used to run the python code of LearningTest and TrueMachine.py. The only library it does not contain is the one that enable us to display graphics on a remote computer (I forgot its name) because it was too big for github.

# processing_result

It contains the processed fire data needed for learning. 
The Australia_fire_xxx files contain 3 .csv files that provide information on the fire and a .png that gives a visual representation of the data.
processed_wind.txt contains information on the wind values for each entry of Australia_fire_xxx.

# config.yaml

config.yaml is the configuration file needed to start the ray cluster with the command `ray up config.yaml`. To make it work on your onw account, make sure to change the paths and to provide a path to your connection key.




### La Classe Cell

Implémentation des cellules de l'automate, contient plusieurs attributs qui indiquent l'état de la cellule :

- `state` : 1 pour enflammable, 2 pour enflammée, 3 pour consommée
- `moisture` : valeur de l'humidité

... et bien d'autre à venir (on se restreint pour le moment à ceux là)

La méthode `transition`, étant donné la liste des cellules voisines et le paramètre de l'automate, renvoie une nouvelle cellule qui représente la cellule à l'instant suivant

### La Classe Automaton

Implémentation de l'automate.

Le constructeur prend en argument : 
- `beta` : le paramètre de l'automate (un vecteur numpy) [Pour le moment, il prend juste la forme de deux paramètres `c_intercept` et `c_moisture`]
- `shape` : la dimension de l'espace à simuler
- `firestart` : la coordonnée du point de départ du feu
- `moisture` : la matrice qui contient la valeur de l'humidité en tout point de l'espace

Les méthodes :
- `get_neighbors` renvoie la liste des cellules voisines (diagonales incluses) de (ci, cj)
- `time_step` actualise l'automate après la simulation d'une étape
- `run` simule l'automate jusqu'à ce que toutes les cellules soient dans l'état 1 ou 3 et renvoie le nombre de cellule brulée [Pour le moment c'est un `print`pour plus de visibilité]

Le reste c'est essentiellement du bazar qui sert à visualiser l'automate mais qui n'est pas fondamentalement important. Il faut noter que `evolution_animation` ne fonctionne pas (et c'est bien dommage mais c'est la vie).


# IdÃ©es

### Descente de gradient

On choisi un paramètre initial au hasard : paramètre = paramètre de régression de p = B vecteur de dimension n + 1 où n est le nombre de variables explicatives (humidité, température, vent selon x, vent selon y, etc...) (Il s'appelle B parce que markdown n'aime pas beta)

- On sélectionne un sous ensemble de l'ensemble d'apprentissage
- Pour chaque instance du sous-ensemble, (zone brulée, carte d'humidité, carte de température, etc...) on construit l'automate associé avec le paramètre actuel B.
- On calcule le gradient pour cette instance en approximant la dérivée au premier ordre : dR/dx_i(B) = (R(B + h*x_i) - R(B))/h pour un petit h. En calculant ce scalaire pour i allant de 0 à n (inclus), on obtient le gradient en B correspondant à cette instance.
- On calcule le gradient moyenné sur toutes les instances du sous-ensemble d'apprentissage
- On fait B = B - r*gradient où r > 0 est le taux d'apprentissage
- On recommence


### Idées d'optimisation

- Réaliser toutes les exécutions de l'automate nécéssaires au calcul du gradient simultanément pour tirer partie de la vectorialisation avec Numpy (à voir expérimentalement, peut-être que ça ne marchera pas, et peut-être que la mémoire ne va pas aider)
- Conserver les valeurs des variables explicatives locales dans des vecteurs plutôt que dans des objets fait main pour exploiter la vitesse de Numpy


### Quelques liens pour classifier les pics (si jamais c'est réelement utile)

- https://www.baeldung.com/cs/clustering-unknown-number
- https://stats.stackexchange.com/questions/217875/clustering-very-small-datasets
- https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318


### Notes pour le rapport (pour se souvenir de ce des difficultés rencontrées et tout)



What happens is that the code of the imported class is only run when it is first imported. If the main code is run again with no change made to imported classes, they are not reloaded and their code is not run again, which might lead to unexpected behaviour with regard to seeds.

Code in imported classes may react unexpectedly to the seed in the main code (is random at first, but after a few call starts following the seed). In any case, all code inside main, regardless of whether it uses an imported class, always follows the seed.

Bigdata is always the same even without seed in machine and data_generator. This is not normal.
This is because perlin sets a seed
We want to know if we can call a function that follows a certain seed and then keep having a random behaviour. The thing is that if we set a seed and then perlin sets another, we want to be able to go back to the original one.
Can we fix this by isolating perlin in a different class ?

We cannot

What we will do then is specify to perlin the seed it should revert to after it has finished running



- Test de résolution du problème du coût aléatoire avec des automates déterministes
- Apprentissage : coût divisé par 10, mais les paramètres ne semblent pas converger vers ce à quoi on s'attend (-12, 18 au lieu de 0,-7 avec valeurs initiales -10,20), et ce même avec le coût défini localement
- Solution potentielle : les données sont en réalité toutes pourries. Plus de 90% des feu ne contiennent qu'une cellule brûlé, ce qui encourage bêtement la machine à prendre des paramètres qui empêchent le feu de commencer.
- Une solution consiste donc à filtrer les échantillon de donnée, et ne conserver que ceux qui ont plus d'une case brulé.
- Ce qui s'est passé en réalité : le fait que les automates deviennent déterministes a limité la taille des feu. En effet, les paramètres du génerateurs étaient calibré pour avoir une taille de feu raisonnable, mais la taille des feu a été diminué puisque le déterminisme à essentiellement fait comme si le nombre d'étape pendant lequel une cellule est en feu est de 1. Ainsi nos feu étaient tous microscopiques.
