# https://stanfordnlp.github.io/stanza/
# http://stanza.run/

import stanza
import sentenza
from sentenza import Display

# CONFIGURATION
# phrases = [
#     "L’invention des arts étant un droit d’aînesse,",
#     "Nous devons l’apologue à l’ancienne Grèce.",
#     "Mais ce champ ne se peut tellement moissonner",
#     "Que les derniers venus n’y trouvent à glaner.",
#     "La feinte est un pays plein de terres désertes.",
#     "Tous les jours nos auteurs y font des découvertes.",
# ]

phrases = [
    "Il était une fois une petite fille de village,",
    "la plus jolie qu'on eût su voir :",
    "sa mère en était folle,",
    "et sa mère-grand plus folle encore.",

    "Cette bonne femme lui fit faire un petit chaperon rouge qui lui seyait si bien,",
    "que partout on l'appelait le petit Chaperon rouge.",

    "Un jour, sa mère ayant cuit et fait des galettes, lui dit :",

    "Va voir comment se porte ta mère-grand,",
    "car on m'a dit qu'elle était malade.",
    "Porte-lui une galette et ce petit pot de beurre.",

    "Le petit Chaperon rouge partit aussitôt pour aller chez sa mère-grand,",
    "qui demeurait dans un autre village.",

    "En passant dans un bois,",
    "elle rencontra compère le Loup,",
    "qui eut bien envie de la manger ;",
    "mais il n'osa, à cause de quelques bûcherons qui étaient dans la forêt.",

    "Il lui demanda où elle allait.",
    "La pauvre enfant,",
    "qui ne savait pas qu'il était dangereux de s'arrêter à écouter un loup, lui dit :",

    "Je vais voir ma mère-grand,",
    "et lui porter une galette,",
    "avec un petit pot de beurre,",
    "que ma mère lui envoie.",

    "Demeure-t-elle bien loin ?",
    "lui dit le Loup.",

    # "Oh! oui, dit le petit Chaperon rouge,",
    "c'est par-delà le moulin que vous voyez tout là-bas,",
    "à la première maison du village.",

    # "Eh bien ! dit le Loup,",
    "je veux l'aller voir aussi ;",
    "je m'y en vais par ce chemin-ci,",
    # "et toi par ce chemin-là ;",
    "et nous verrons à qui plus tôt y sera.",

    "Le Loup se mit à courir de toute sa force par le chemin qui était le plus court,",
    "et la petite fille s'en alla par le chemin le plus long,",
    "s'amusant à cueillir des noisettes,",
    "à courir après des papillons,",
    "et à faire des bouquets des petites fleurs qu'elle rencontrait.",

    "Le Loup ne fut pas longtemps à arriver à la maison de la mère-grand ;",
    "il heurte : toc, toc.",

    "- Qui est là ?",
    "- C'est votre fille,",
    "le petit Chaperon rouge,",
    "dit le Loup en contrefaisant sa voix,",
    "qui vous apporte une galette et un petit pot de beurre, que ma mère vous envoie.",

    "La bonne mère-grand,",
    "qui était dans son lit,",
    # "à cause qu'elle se trouvait un peu mal,",
    "lui cria :",

    "Tire la chevillette,",
    "la bobinette cherra.",

    "Le Loup tira la chevillette,",
    "et la porte s'ouvrit.",
    "Il se jeta sur la bonne femme,",
    "et la dévora en moins de rien,",
    "car il y avait plus de trois jours qu'il n'avait mangé.",
    "Ensuite il ferma la porte,",
    "et s'en alla coucher dans le lit de la mère-grand,",
    "en attendant le petit Chaperon rouge,",
    # "qui, quelque temps après,",
    "vint heurter à la porte : toc, toc :",

    "Qui est là ?",

    "Le petit Chaperon rouge,",
    "qui entendit la grosse voix du Loup,",
    "eut peur d'abord,",
    "mais croyant que sa mère-grand était enrhumée,",
    "répondit :",

    "C'est votre fille,",
    "le petit Chaperon rouge,",
    "qui vous apporte une galette et un petit pot de beurre,",
    "que ma mère vous envoie.",

    "Le Loup lui cria en adoucissant un peu sa voix :",
    "Tire la chevillette, la bobinette cherra.",

    "Le petit Chaperon rouge tira la chevillette, et la porte s'ouvrit.",

    "Le Loup, la voyant entrer,",
    "lui dit en se cachant dans le lit,",
    "sous la couverture :",

    "Mets la galette et le petit pot de beurre sur la huche,",
    "et viens te coucher avec moi.",

    "Le petit Chaperon rouge se déshabille,",
    "et va se mettre dans le lit,",
    "où elle fut bien étonnée de voir comment sa mère-grand était faite en son déshabillé. Elle lui dit :",

    "- Ma mère-grand, que vous avez de grands bras !",
    "- C'est pour mieux t'embrasser, ma fille !",
    "- Ma mère-grand, que vous avez de grandes jambes !",
    "- C est pour mieux courir, mon enfant !",
    "- Ma mère-grand, que vous avez de grandes oreilles !",
    "- C'est pour mieux écouter, mon enfant !",
    "- Ma mère-grand, que vous avez de grands yeux !",
    "- C'est pour mieux te voir, mon enfant !",
    "- Ma mère-grand, que vous avez de grandes dents !",
    "- C'est pour te manger !",

    "Et, en disant ces mots, ce méchant Loup se jeta sur le petit Chaperon rouge, et la mangea.",
]

language = "Fr"
display = Display.GLOSSES

# OBJET DOCUMENT CREATION
stanza.download(language)
nlp = stanza.Pipeline(language)  # initialize neural pipeline

if display == Display.GLOSSES:

    # GLOSSES USAGE
    for phrase in phrases:
        print(phrase+" : \n")
        list_of_linked_glosses = sentenza.load_sentenza(nlp(phrase))  # create a WordTree
        list_of_linked_glosses.display_gloss_from_inner_tree(list_of_linked_glosses.items[0])
        list_of_linked_glosses.add_orphans()
        list_of_linked_glosses.write()
        print("---\n")

else:

    # DISPLAY LIST USAGE
    for phrase in phrases:
        print(phrase+" : \n")
        list_of_linked_glosses = sentenza.load_sentenza(nlp(phrase))
        list_of_linked_glosses.show(list_of_linked_glosses.items[0], "")
        print("---\n")
