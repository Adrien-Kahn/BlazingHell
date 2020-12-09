# Documentation rapide

### La Classe Cell

Impl�mentation des cellules de l'automate, contient plusieurs attributs qui indiquent l'�tat de la cellule :

`state`: 1 pour enflammable, 2 pour enflamm�e, 3 pour consomm�e
`moisture`: valeur de l'humidit�
...

### La Classe Automata

Impl�mentation de l'automate.



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
- R�aliser toutes les execution de l'automate n�c�ssaire au calcul du gradient simultan�ment pour tirer partie de la vectorialisation avec Numpy (� voir exp�rimentalement, peut-�tre que �a ne marchera pas, et peut-�tre que la m�moire ne va pas aider)