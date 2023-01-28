# https://stanfordnlp.github.io/stanza/
# http://stanza.run/

# todo camembert.py (+ try names from everywhere in the world)

import stanza
import sentenza

# 1. Temps (évènement)
# 2. Lieu (classificateurs)
# 3. Personnages (ce dont on parle, classificateurs)
# 4. Action (verbe)

# Data snooping inside!!
# Is discovering a meteo subset will generalize?

# CONFIGURATION
# phrases = [
#     "Des anticyclones viennent du Nord",
#     "Est-ce que Poutine est le complément du green washing?",
#     "Le soleil a brillé tout le mois de Mars mais ...",
#     "Sortez vos parapluies, il va bientôt pleuvoir",
#     "La carte de la Belgique est cachée par un gros nuage",
#     "Ce mardi, il fera 12 degrés en matinée",
#     "La nuit nous descendrons en dessous de zéro",
#     "Demain, nous aurons des températures douces!",
#     "Les hirondelles aiment danser au dessus de la pluie."
# ]

phrases = [
    "Ce matin, il y aura un beau soleil",
    "La température ce matin sera de 8 degrés",
    "La force du vent oscillera aux allentours de de neuf km/h",
    "L'humidité relative de l'air sera de 89 pourcents."
]

language = "Fr"

# OBJET DOCUMENT CREATION
stanza.download(language)
nlp = stanza.Pipeline(language)  # initialize neural pipeline

# TREE USAGE
for phrase in phrases:
    print(phrase+" : \n")
    tree = sentenza.load_sentenza(nlp(phrase))
    tree.show(tree.nodes[0], "")
    # tree.gloss(tree.nodes[0])
    print("---\n")

