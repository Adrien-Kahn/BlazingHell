Salut, petit test, nous nous trouvons bien dans la branche cache-memo

# Documentation rapide

### La Classe Cell

Impl�mentation des cellules de l'automate, contient plusieurs attributs qui indiquent l'�tat de la cellule :

- `state` : 1 pour enflammable, 2 pour enflamm�e, 3 pour consomm�e
- `moisture` : valeur de l'humidit�

... et bien d'autre � venir (on se restreint pour le moment � ceux l�)

La m�thode `transition`, �tant donn� la liste des cellules voisines et le param�tre de l'automate, renvoie une nouvelle cellule qui repr�sente la cellule � l'instant suivant

### La Classe Automaton

Impl�mentation de l'automate.

Le constructeur prend en argument : 
- `beta` : le param�tre de l'automate (un vecteur numpy) [Pour le moment, il prend juste la forme de deux param�tres `c_intercept` et `c_moisture`]
- `shape` : la dimension de l'espace � simuler
- `firestart` : la coordonn�e du point de d�part du feu
- `moisture` : la matrice qui contient la valeur de l'humidit� en tout point de l'espace

Les m�thodes :
- `get_neighbors` renvoie la liste des cellules voisines (diagonales incluses) de (ci, cj)
- `time_step` actualise l'automate apr�s la simulation d'une �tape
- `run` simule l'automate jusqu'� ce que toutes les cellules soient dans l'�tat 1 ou 3 et renvoie le nombre de cellule brul�e [Pour le moment c'est un `print`pour plus de visibilit�]

Le reste c'est essentiellement du bazar qui sert � visualiser l'automate mais qui n'est pas fondamentalement important. Il faut noter que `evolution_animation` ne fonctionne pas (et c'est bien dommage mais c'est la vie).


# Id�es

### Descente de gradient

On choisi un param�tre initial au hasard : param�tre = param�tre de r�gression de p = B vecteur de dimension n + 1 o� n est le nombre de variables explicatives (humidit�, temp�rature, vent selon x, vent selon y, etc...) (Il s'appelle B parce que markdown n'aime pas beta)

- On s�lectionne un sous ensemble de l'ensemble d'apprentissage
- Pour chaque instance du sous-ensemble, (zone brul�e, carte d'humidit�, carte de temp�rature, etc...) on construit l'automate associ� avec le param�tre actuel B.
- On calcule le gradient pour cette instance en approximant la d�riv�e au premier ordre : dR/dx_i(B) = (R(B + h*x_i) - R(B))/h pour un petit h. En calculant ce scalaire pour i allant de 0 � n (inclus), on obtient le gradient en B correspondant � cette instance.
- On calcule le gradient moyenn� sur toutes les instances du sous-ensemble d'apprentissage
- On fait B = B - r*gradient o� r > 0 est le taux d'apprentissage
- On recommence


### Id�es d'optimisation

- Conserver en cache les cellules "actives", c'est-�-dire celles enflamm�e ou enflammable voisines d'enflamm�e, pour significativement r�duire le nombre de cellules trait�s � chaque �tape et assurer une complexit� d'execution de l'automate de l'ordre de O(kn�)
- R�aliser toutes les ex�cutions de l'automate n�c�ssaires au calcul du gradient simultan�ment pour tirer partie de la vectorialisation avec Numpy (� voir exp�rimentalement, peut-�tre que �a ne marchera pas, et peut-�tre que la m�moire ne va pas aider)
- Conserver les valeurs des variables explicatives locales dans des vecteurs plut�t que dans des objets fait main pour exploiter la vitesse de Numpy