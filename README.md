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

On choisi des param�tres initiaux au hasard (param�tre = param�tres de r�gr�ssion de $p$)

- On s�lectionne un sous ensemble de l'ensemble d'apprentissage
- Pour chaque instance du sous-ensemble, (zone brul�e, carte humidit�, carte temp�rature, etc...) on construit l'automate associ� avec les param�tres actuels