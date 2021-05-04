# What you will find in this repository

### `DataCollectionAndProcessing`

`DataCollectionAndProcessing` contains scripts related to the collection and processing of fire data.
`firedata_automation.py` is the script used to extract and process fire data from Copernicus.

### 'DeterministicModel`

`DeterministicModel` contains the implementation of the model described in 5.4

### `LearningTest`

`LearningTest` contains the implementation of the first model described from 5.1 to 5.3 and in 5.5.

## `Automata.py`

`Automata.py` implements the basic data structures needed to work with the inital model, namely `Cell` and `Automaton`.

# The class `Cell`

`Cell` is, as explained in the report, the description of a cell of the automata. It simply contains all the information in that cell :
- `moisture` : the value of moisture for that cell
- `fuel` : the remaining fuel in that cell
- `state` : the current state of the cell

The state of the cell is coded by a integer :
- 1 : flammable
- 2 : burning
- 3 : burnt


# The class Automaton

`Automaton` is the class that implements the cellular automaton. Its fields are :

- `mat` a matrix of `Cell` that represents the current state of the automaton
- `c_intercept` the constant coefficient
- `c_moisture` the coefficient associated to moisture


- `ii` and `jj` the shape of the automaton's grid




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


# Idées

### Descente de gradient

On choisi un param�tre initial au hasard : param�tre = param�tre de r�gression de p = B vecteur de dimension n + 1 o� n est le nombre de variables explicatives (humidit�, temp�rature, vent selon x, vent selon y, etc...) (Il s'appelle B parce que markdown n'aime pas beta)

- On s�lectionne un sous ensemble de l'ensemble d'apprentissage
- Pour chaque instance du sous-ensemble, (zone brul�e, carte d'humidit�, carte de temp�rature, etc...) on construit l'automate associ� avec le param�tre actuel B.
- On calcule le gradient pour cette instance en approximant la d�riv�e au premier ordre : dR/dx_i(B) = (R(B + h*x_i) - R(B))/h pour un petit h. En calculant ce scalaire pour i allant de 0 � n (inclus), on obtient le gradient en B correspondant � cette instance.
- On calcule le gradient moyenn� sur toutes les instances du sous-ensemble d'apprentissage
- On fait B = B - r*gradient o� r > 0 est le taux d'apprentissage
- On recommence


### Id�es d'optimisation

- R�aliser toutes les ex�cutions de l'automate n�c�ssaires au calcul du gradient simultan�ment pour tirer partie de la vectorialisation avec Numpy (� voir exp�rimentalement, peut-�tre que �a ne marchera pas, et peut-�tre que la m�moire ne va pas aider)
- Conserver les valeurs des variables explicatives locales dans des vecteurs plut�t que dans des objets fait main pour exploiter la vitesse de Numpy


### Quelques liens pour classifier les pics (si jamais c'est r�element utile)

- https://www.baeldung.com/cs/clustering-unknown-number
- https://stats.stackexchange.com/questions/217875/clustering-very-small-datasets
- https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318


### Notes pour le rapport (pour se souvenir de ce des difficult�s rencontr�es et tout)



What happens is that the code of the imported class is only run when it is first imported. If the main code is run again with no change made to imported classes, they are not reloaded and their code is not run again, which might lead to unexpected behaviour with regard to seeds.

Code in imported classes may react unexpectedly to the seed in the main code (is random at first, but after a few call starts following the seed). In any case, all code inside main, regardless of whether it uses an imported class, always follows the seed.

Bigdata is always the same even without seed in machine and data_generator. This is not normal.
This is because perlin sets a seed
We want to know if we can call a function that follows a certain seed and then keep having a random behaviour. The thing is that if we set a seed and then perlin sets another, we want to be able to go back to the original one.
Can we fix this by isolating perlin in a different class ?

We cannot

What we will do then is specify to perlin the seed it should revert to after it has finished running



- Test de r�solution du probl�me du co�t al�atoire avec des automates d�terministes
- Apprentissage : co�t divis� par 10, mais les param�tres ne semblent pas converger vers ce � quoi on s'attend (-12, 18 au lieu de 0,-7 avec valeurs initiales -10,20), et ce m�me avec le co�t d�fini localement
- Solution potentielle : les donn�es sont en r�alit� toutes pourries. Plus de 90% des feu ne contiennent qu'une cellule br�l�, ce qui encourage b�tement la machine � prendre des param�tres qui emp�chent le feu de commencer.
- Une solution consiste donc � filtrer les �chantillon de donn�e, et ne conserver que ceux qui ont plus d'une case brul�.
- Ce qui s'est pass� en r�alit� : le fait que les automates deviennent d�terministes a limit� la taille des feu. En effet, les param�tres du g�nerateurs �taient calibr� pour avoir une taille de feu raisonnable, mais la taille des feu a �t� diminu� puisque le d�terminisme � essentiellement fait comme si le nombre d'�tape pendant lequel une cellule est en feu est de 1. Ainsi nos feu �taient tous microscopiques.



kill.sh



cat liste.txt | while read line || [[ -n $line ]];

do

   sshpass -p motdepasse ssh -oStrictHostKeyChecking=no -f maxime.peim@$line.polytechnique.fr "export LC_ALL=fr_FR.utf8; source /users/eleves-b/2018/maxime.peim/INF568/Lab2/venv/bin/activate; ray stop --force"

done





liste.txt



bentley

bugatti

cadillac

chrysler

corvette

ferrari

fiat

ford

jaguar

lada

maserati

mazda

nissan

niva

peugeot

pontiac

porsche

renault

rolls

rover

royce

simca

skoda

venturi

volvo

albatros

autruche

bengali

coucou

dindon

epervier

faisan

gelinotte

harpie

hibou

jabiru

kamiche

linotte

loriol

mouette

nandou

ombrette

perdrix

quetzal

quiscale

rouloul

sitelle

traquet

urabu

verdier

acromion

apophyse

astragale

atlas

axis

coccyx

cote

cubitus

cuboide

femur

frontal

humerus

malleole

metacarpe

parietal

perone

phalange

radius

rotule

sacrum

sternum

tarse

temporal

tibia

xiphoide

aerides

barlia

calanthe

diuris

encyclia

epipactis

gennaria

habenaria

isotria

ipsea

liparis

lycaste

malaxis

neotinea

oncidium

ophrys

orchis

pleione

pogonia

serapias

telipogon

vanda

vanilla

xylobium

zeuxine

