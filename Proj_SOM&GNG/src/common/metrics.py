import numpy as np


def erro_quantizacao(rede, pixels):
    pixels = np.asarray(pixels, dtype=np.float64)
    bmu1, _ = rede.encontrar_bmus(pixels)
    return _eq_a_partir_dos_bmus(pixels, rede.obter_pesos(), bmu1)


def erro_topologico(rede, pixels):
    pixels = np.asarray(pixels, dtype=np.float64)
    bmu1, bmu2 = rede.encontrar_bmus(pixels)
    return _et_a_partir_dos_bmus(rede, bmu1, bmu2)


def avaliar_rede(rede, pixels):
    pixels = np.asarray(pixels, dtype=np.float64)
    bmu1, bmu2 = rede.encontrar_bmus(pixels)
    return {
        "erro_quantizacao": _eq_a_partir_dos_bmus(pixels, rede.obter_pesos(), bmu1),
        "erro_topologico": _et_a_partir_dos_bmus(rede, bmu1, bmu2),
    }

def _eq_a_partir_dos_bmus(pixels, pesos, bmu1):
    pesos = np.asarray(pesos, dtype=np.float64)
    # pesos[bmu1] = a cor do neurônio vencedor de cada pixel -> (N, 3)
    distancias = np.linalg.norm(pixels - pesos[bmu1], axis=1)
    return float(distancias.mean())


def _et_a_partir_dos_bmus(rede, bmu1, bmu2):
    # Pares (1º, 2º) distintos e quantos pixels caíram em cada um
    pares, contagens = np.unique(np.stack([bmu1, bmu2], axis=1), axis=0, return_counts=True)

    pixels_sem_vizinhanca = 0
    for (i, j), qtd in zip(pares, contagens):
        if not rede.sao_vizinhos(int(i), int(j)):
            pixels_sem_vizinhanca += int(qtd)

    return pixels_sem_vizinhanca / len(bmu1)