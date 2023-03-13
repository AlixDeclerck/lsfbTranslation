# https://stanfordnlp.github.io/stanza/
# http://stanza.run/

import stanza
from algorithms.syntax_analysis.with_stanza import sentenza
from data import conte

# CONFIGURATION
phrases = [
    "La perfection réside dans l'amour que l'on porte.",
    "Chaque nouvelle journée est une instance unique comme les autres.",
    "La définition du mot femme est aussi étendue que le coeur des humains sait s'ouvrir.",
    "Je considère que Edward Snowden est un héros et le fait qu'il n'ai trouvé asile qu'en Russie me terrifie!",
]

language = "Fr"
stanza.download(language)
nlp = stanza.Pipeline(language)

for phrase in conte.the_prince_frog:
    print(phrase+" : \n")

    # create a WordTree
    glosses = sentenza.load_sentenza(nlp(phrase))
    glosses.list_display()

    print("---\n")
