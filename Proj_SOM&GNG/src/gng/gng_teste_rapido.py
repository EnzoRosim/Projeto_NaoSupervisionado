"""
gng_teste_rapido.py
-------------------
Script TEMPORÁRIO para ver a GNG funcionando em uma imagem, antes de existir
o `experiments`. Funciona de qualquer pasta (PyCharm, terminal...): ele descobre
sozinho onde está a raiz do projeto (a pasta que contém `database/`).

    python src/gng/gng_teste_rapido.py                       (usa database/image1.png)
    python src/gng/gng_teste_rapido.py database/image3.png   (usa outra imagem)

Treina as 3 GNGs (16, 64 e 256 neurônios), imprime os erros e salva em
output/graphics/gng/<imagem>/ as figuras de comparação e de métricas.
"""

import sys
import time
from pathlib import Path


def achar_raiz():
    """Sobe a partir deste arquivo até achar a pasta que contém `database/`."""
    for pasta in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (pasta / "database").is_dir():
            return pasta
    raise FileNotFoundError(
        "Não achei a pasta `database/` acima deste arquivo. "
        "Confira se o projeto tem a pasta database na raiz."
    )


RAIZ = achar_raiz()
sys.path.insert(0, str(RAIZ))   # permite `from src...` mesmo rodando de outra pasta

from src.common.image_utils import (  # noqa: E402
    carregar_imagem, imagem_para_pixels, reconstruir_imagem, paleta_unica, listar_imagens,
)
from src.common.metrics import avaliar_rede  # noqa: E402
from src.common.plots import plotar_comparacao, plotar_metricas_imagem  # noqa: E402
from src.gng.gng_configs import criar_todas_gng  # noqa: E402

LADO_MAXIMO = 128   # reduz a imagem para o maior lado ter no máximo 128 px


def resolver_caminho():
    """Caminho da imagem: argumento (relativo à raiz do projeto) ou database/image1.png."""
    texto = sys.argv[1] if len(sys.argv) > 1 else "database/image5.png"
    caminho = Path(texto)
    if not caminho.is_absolute():
        caminho = RAIZ / caminho

    if not caminho.is_file():
        print(f"Imagem não encontrada: {caminho}")
        print(f"Imagens que existem em {RAIZ / 'database'}:")
        for p in listar_imagens(RAIZ / "database"):
            print(f"  - {p.name}")
        sys.exit(1)
    return caminho


def main():
    caminho = resolver_caminho()
    nome_img = caminho.stem                                   # "image1"
    pasta_saida = RAIZ / "output" / "graphics" / "gng" / nome_img

    imagem = carregar_imagem(caminho, lado_maximo=LADO_MAXIMO)
    pixels = imagem_para_pixels(imagem)
    print(f"{nome_img}: {imagem.shape[1]}x{imagem.shape[0]} px, "
          f"{len(pixels)} pixels, {paleta_unica(imagem)} cores distintas\n")

    resultados = []
    for rede in criar_todas_gng():
        t0 = time.time()
        rede.treinar(pixels)
        segundos = time.time() - t0

        metricas = avaliar_rede(rede, pixels)
        resultados.append({"tipo": rede.tipo, "arquitetura": rede.arquitetura, **metricas})

        reconstruida = reconstruir_imagem(rede.quantizar(pixels), rede.obter_pesos(), imagem.shape)
        plotar_comparacao(
            imagem, reconstruida, rede.obter_pesos(),
            pasta_saida / f"{nome_img}_{rede.nome}.png",
            titulo=f"{nome_img} - GNG {rede.arquitetura} "
                   f"({rede.n_neuronios} neurônios, {rede.n_arestas} arestas)",
        )
        print(f"GNG {rede.arquitetura:>4}: {rede.n_neuronios:>3} neurônios, {rede.n_arestas:>4} arestas | "
              f"erro quantização = {metricas['erro_quantizacao']:.4f} | "
              f"erro topológico = {metricas['erro_topologico']:.4f} | {segundos:.1f}s")

    plotar_metricas_imagem(resultados, pasta_saida / f"{nome_img}_gng_metricas.png",
                           titulo=f"{nome_img} - GNG")
    print(f"\nFiguras salvas em {pasta_saida}")


if __name__ == "__main__":
    main()