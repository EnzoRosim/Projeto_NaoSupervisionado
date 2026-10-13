from pathlib import Path

import numpy as np
from PIL import Image

# Extensões aceitas ao listar a pasta database/ utilizaremos .png como base
EXTENSOES_VALIDAS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ----------------------------------------------------------------------
# 1) EXTRAÇÃO: da imagem para os dados que a rede vai usar
# ----------------------------------------------------------------------

def listar_imagens(pasta_database):
    #Lista as imagens de uma pasta, em ordem alfabética (image1, image2, ...).

    pasta = Path(pasta_database)
    if not pasta.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {pasta}")

    imagens = [p for p in pasta.iterdir() if p.suffix.lower() in EXTENSOES_VALIDAS]
    return sorted(imagens)


def carregar_imagem(caminho, lado_maximo=None):
    #Abre uma imagem colorida e devolve um array (altura, largura, 3) uint8.

    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Imagem não encontrada: {caminho}")

    imagem = Image.open(caminho).convert("RGB")

    if lado_maximo is not None and max(imagem.size) > lado_maximo:
        # thumbnail reduz mantendo a proporção (e nunca aumenta a imagem)
        imagem.thumbnail((lado_maximo, lado_maximo), Image.LANCZOS)

    return np.asarray(imagem, dtype=np.uint8)


def imagem_para_pixels(imagem):
    """
    Transforma a imagem (H, W, 3) numa lista de pixels (N, 3) normalizada em [0, 1].

    Cada linha da saída é uma amostra com 3 características: [R, G, B].

    Exemplo simples (imagem 2x2):
        imagem = [[[255, 0, 0], [0, 255, 0]],
                  [[0, 0, 255], [255, 255, 255]]]
        pixels = [[1, 0, 0],      <- vermelho
                  [0, 1, 0],      <- verde
                  [0, 0, 1],      <- azul
                  [1, 1, 1]]      <- branco
    """
    _validar_imagem(imagem)
    pixels = imagem.reshape(-1, 3).astype(np.float64) / 255.0
    return pixels


# ----------------------------------------------------------------------
# 4) RECONSTRUÇÃO: da saída da rede de volta para uma imagem
# ----------------------------------------------------------------------

def reconstruir_imagem(bmu_idx, pesos, forma_original):
    """
    Monta a imagem quantizada: cada pixel recebe a cor do seu neurônio vencedor.

    Parâmetros:
        bmu_idx        : (N,) índice do neurônio vencedor de cada pixel
        pesos          : (K, 3) cores aprendidas pela rede, em [0, 1]
        forma_original : (altura, largura, 3), vem de `imagem.shape`

    Exemplo simples:
        pesos   = [[1, 0, 0],   # neurônio 0 = vermelho
                   [0, 0, 1]]   # neurônio 1 = azul
        bmu_idx = [0, 1, 1, 0]  # 4 pixels
        -> pixels reconstruídos = [vermelho, azul, azul, vermelho]
        -> reorganizados em 2x2 pela `forma_original`.
    """
    bmu_idx = np.asarray(bmu_idx)
    pesos = np.asarray(pesos, dtype=np.float64)

    if pesos.ndim != 2 or pesos.shape[1] != 3:
        raise ValueError(f"`pesos` deve ter forma (K, 3), mas veio {pesos.shape}")

    altura, largura = forma_original[0], forma_original[1]
    if bmu_idx.shape != (altura * largura,):
        raise ValueError(
            f"`bmu_idx` deve ter {altura * largura} elementos, mas veio {bmu_idx.shape}"
        )
    if bmu_idx.min() < 0 or bmu_idx.max() >= len(pesos):
        raise ValueError("`bmu_idx` contém índice de neurônio inexistente")

    # Truque do NumPy: pesos[bmu_idx] pega, de uma vez, a linha de `pesos`
    # indicada por cada índice. Resultado: array (N, 3).
    pixels_novos = pesos[bmu_idx]
    return pixels_para_imagem(pixels_novos, forma_original)


def pixels_para_imagem(pixels, forma_original):
    """
    Operação inversa de `imagem_para_pixels`: (N, 3) em [0, 1] -> (H, W, 3) uint8.

    O `np.clip` protege contra valores que fujam um pouco de [0, 1].
    O `np.round` evita que 0.999 vire 254 (por truncamento) em vez de 255.
    """
    pixels = np.clip(np.asarray(pixels, dtype=np.float64), 0.0, 1.0)
    imagem = np.round(pixels * 255.0).astype(np.uint8)
    return imagem.reshape(forma_original)


def paleta_unica(imagem):
    """
    Conta quantas cores DISTINTAS existem na imagem.
    Útil para mostrar o "antes e depois" da quantização (ex.: 41.230 -> 64 cores).
    """
    _validar_imagem(imagem)
    return len(np.unique(imagem.reshape(-1, 3), axis=0))


# ----------------------------------------------------------------------
# Salvar
# ----------------------------------------------------------------------

def salvar_imagem(imagem, caminho):
    """
    Salva a imagem (H, W, 3) uint8 em disco, criando as pastas se não existirem.

    Exemplo:
        salvar_imagem(img, "output/graphics/som/image3/image3_som_8x8.png")
    """
    _validar_imagem(imagem)
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(imagem, mode="RGB").save(caminho)


# ----------------------------------------------------------------------
# Auxiliar interna
# ----------------------------------------------------------------------

def _validar_imagem(imagem):
    """Confere se `imagem` segue a convenção (H, W, 3) uint8."""
    if not isinstance(imagem, np.ndarray):
        raise TypeError("A imagem deve ser um np.ndarray")
    if imagem.ndim != 3 or imagem.shape[2] != 3:
        raise ValueError(f"A imagem deve ter forma (H, W, 3), mas veio {imagem.shape}")
    if imagem.dtype != np.uint8:
        raise TypeError(f"A imagem deve ser uint8, mas veio {imagem.dtype}")