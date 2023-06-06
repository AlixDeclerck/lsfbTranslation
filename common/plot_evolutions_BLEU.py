import numpy as np
import matplotlib.pyplot as plt
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config

scores_BLEU = [57.33, 58.7, 59.84]
N = [1000, 2000, 3000]

if __name__ == '__main__':

    config = load_config("../algorithms/symbolicTransformer/src/config.yaml")

    m_bleu, b_bleu = np.polyfit(N, scores_BLEU, deg=1)
    title_regression2 = "régression sur le score BLEU / taille du vocabulaire"
    BLEU_score = "Erreurs mesurées avec un score BLEU lors de la session 5"

    plt.plot(N, scores_BLEU, "bo", color=str(config["graphics"]["color2"]), label=BLEU_score)
    plt.axline(xy1=(0, b_bleu), slope=m_bleu, label=title_regression2, color=str(config["graphics"]["color4"]))

    plt.ylim(0, 100)
    plt.xlim(0, 50000)
    plt.legend()

    plt.show()
