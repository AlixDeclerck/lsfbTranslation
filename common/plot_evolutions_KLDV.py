import numpy as np
import matplotlib.pyplot as plt
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config

scores_BLEU = [57.33, 58.7, 59.84]
scores_KLDV = [2410, 2327, 2303]
N = [1000, 2000, 3000]

if __name__ == '__main__':

    config = load_config("../algorithms/symbolicTransformer/src/config.yaml")

    m, b = np.polyfit(N, scores_KLDV, deg=1)
    m_bleu, b_bleu = np.polyfit(N, scores_BLEU, deg=1)
    title_regression = "régression sur l'erreur / taille du vocabulaire"
    title_regression2 = "régression sur le score BLEU / taille du vocabulaire"
    error_score = "Erreurs mesurées en validation lors de la session 5"
    BLEU_score = "Erreurs mesurées avec un score BLEU lors de la session 5"
    res = f"y = {m:.1f}x {b:+.1f}"

    plt.plot(N, scores_KLDV, "bo", color=str(config["graphics"]["color1"]), label=error_score)
    plt.axline(xy1=(0, b), slope=m, label=title_regression, color=str(config["graphics"]["color3"]))

    plt.plot(N, scores_BLEU, "bo", color=str(config["graphics"]["color2"]), label=BLEU_score)
    plt.axline(xy1=(0, b_bleu), slope=m_bleu, label=title_regression2, color=str(config["graphics"]["color4"]))

    plt.ylim(0, 3000)
    plt.xlim(0, 50000)
    plt.legend()

    plt.show()
