# Documentation rapide

### La Classe Cell

Implémentation des cellules de l'automate, contient plusieurs attributs qui indiquent l'état de la cellule :

`state`: 1 pour enflammable, 2 pour enflammée, 3 pour consommée
`moisture`: valeur de l'humidité
...

### La Classe Automata

Implémentation de l'automate.



# Idées

### Descente de gradient

On choisi des paramètres initiaux au hasard (paramètre = paramètres de régréssion de $p$)

- On sélectionne un sous ensemble de l'ensemble d'apprentissage
- Pour chaque instance du sous-ensemble, (zone brulée, carte humidité, carte température, etc...) on construit l'automate associé avec les paramètres actuels