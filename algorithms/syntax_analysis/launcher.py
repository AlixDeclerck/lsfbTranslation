# https://stanfordnlp.github.io/stanza/
# http://stanza.run/

import stanza
import sentenza
from sentenza import Display

# CONFIGURATION
phrases = [
    "L’invention des arts étant un droit d’aînesse,",
    "Nous devons l’apologue à l’ancienne Grèce.",
    "Mais ce champ ne se peut tellement moissonner",
    "Que les derniers venus n’y trouvent à glaner.",
    "La feinte est un pays plein de terres désertes.",
    "Tous les jours nos auteurs y font des découvertes.",
]

language = "Fr"
display = Display.SHOW_LIST

# OBJET DOCUMENT CREATION
stanza.download(language)
nlp = stanza.Pipeline(language)  # initialize neural pipeline

if display == Display.GLOSSES:

    # GLOSSES USAGE
    for phrase in phrases:
        print(phrase+" : \n")
        list_of_linked_glosses = sentenza.load_sentenza(nlp(phrase))
        list_of_linked_glosses.gloss(list_of_linked_glosses.items[0])
        print("---\n")

else:

    # DISPLAY LIST USAGE
    for phrase in phrases:
        print(phrase+" : \n")
        list_of_linked_glosses = sentenza.load_sentenza(nlp(phrase))
        list_of_linked_glosses.show(list_of_linked_glosses.items[0], "")
        print("---\n")
