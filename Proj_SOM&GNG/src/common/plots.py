from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Cores fixas por tipo de rede, para SOM e GNG ficarem sempre iguais nos gráficos
COR_TIPO = {"som": "#1f77b4", "gng": "#ff7f0e"}


# ----------------------------------------------------------------------
# 1) Original | Quantizada | Paleta   (o gráfico mostrado no enunciado)
# ----------------------------------------------------------------------

def plotar_comparacao(original, quantizada, pesos, caminho, titulo=None, mostrar=False):
    pesos = np.asarray(pesos, dtype=np.float64)

    fig = plt.figure(figsize=(12, 4.5))
    grade = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.6])

    ax1 = fig.add_subplot(grade[0, 0])
    ax1.imshow(original)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2 = fig.add_subplot(grade[0, 1])
    ax2.imshow(quantizada)
    ax2.set_title(f"Quantizada (K={len(pesos)})")
    ax2.axis("off")

    ax3 = fig.add_subplot(grade[0, 2])
    _desenhar_paleta(ax3, pesos)
    ax3.set_title("Paleta aprendida")

    if titulo:
        fig.suptitle(titulo, fontsize=13)
    fig.tight_layout()
    return _finalizar(fig, caminho, mostrar)


def plotar_paleta(pesos, caminho, titulo=None, mostrar=False):
    """Figura só com as cores aprendidas pela rede (um quadradinho por neurônio)."""
    pesos = np.asarray(pesos, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(5, 5))
    _desenhar_paleta(ax, pesos)
    ax.set_title(titulo or f"Paleta aprendida (K={len(pesos)})")
    fig.tight_layout()
    return _finalizar(fig, caminho, mostrar)


# ----------------------------------------------------------------------
# 2) Métricas das redes de uma imagem
# ----------------------------------------------------------------------

def plotar_metricas_imagem(resultados, caminho, titulo=None, mostrar=False):
    if not resultados:
        raise ValueError("`resultados` está vazio")

    rotulos = [f"{r['tipo'].upper()}\n{r['arquitetura']}" for r in resultados]
    cores = [COR_TIPO.get(r["tipo"], "gray") for r in resultados]
    posicoes = np.arange(len(resultados))

    fig, (ax_q, ax_t) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, chave, nome in (
        (ax_q, "erro_quantizacao", "Erro de quantização (menor = melhor)"),
        (ax_t, "erro_topologico", "Erro topológico (menor = melhor)"),
    ):
        valores = [r[chave] for r in resultados]
        barras = ax.bar(posicoes, valores, color=cores)
        ax.bar_label(barras, fmt="%.3f", fontsize=8)
        ax.set_xticks(posicoes)
        ax.set_xticklabels(rotulos, fontsize=9)
        ax.set_title(nome)
        ax.margins(y=0.15)
        ax.spines[["top", "right"]].set_visible(False)

    if titulo:
        fig.suptitle(titulo, fontsize=13)
    fig.tight_layout()
    return _finalizar(fig, caminho, mostrar)


# ----------------------------------------------------------------------
# Auxiliares internas
# ----------------------------------------------------------------------

def _desenhar_paleta(ax, pesos):
    k = len(pesos)
    lado = int(np.ceil(np.sqrt(k)))
    mosaico = np.ones((lado * lado, 3))           # começa tudo branco
    mosaico[:k] = np.clip(pesos, 0.0, 1.0)
    ax.imshow(mosaico.reshape(lado, lado, 3), interpolation="nearest")
    ax.axis("off")


def _finalizar(fig, caminho, mostrar):
    """Salva a figura (criando pastas), mostra se pedido e fecha."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    if mostrar:
        plt.show()
    plt.close(fig)
    return caminho