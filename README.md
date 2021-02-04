# Documentation rapide

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


# Idées

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


### Quelques liens pour classifier les pics

- https://www.baeldung.com/cs/clustering-unknown-number
- https://stats.stackexchange.com/questions/217875/clustering-very-small-datasets
- https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318