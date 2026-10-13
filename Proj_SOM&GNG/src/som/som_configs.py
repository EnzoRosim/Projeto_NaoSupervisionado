"""
som_configs.py
--------------
Arquiteturas e hiperparâmetros do SOM em UM só lugar.
Para ajustar algo (o enunciado pede "ajustes nos hiperparâmetros"), mexa aqui,
sem precisar tocar no modelo.

Arquiteturas pedidas no enunciado: grids 4x4, 8x8 e 16x16.
"""

from src.som.som_model import SOM

# Hiperparâmetros comuns a todas as arquiteturas
CONFIG_BASE = {
    "taxa_inicial": 0.5,
    "taxa_final": 0.01,
    "raio_final": 0.5,
    "semente": 42,
}

# Grids do enunciado: (linhas, colunas)
GRIDS = [(4, 4), (8, 8), (16, 16)]

# Quantas iterações de treino por neurônio.
# Exemplo: 4x4 = 16 neurônios -> 16 * 500 = 8.000 iterações
#          16x16 = 256 neurônios -> 256 * 500 = 128.000 iterações
# Grids maiores têm mais neurônios para "ensinar", então precisam de mais passos.
ITERACOES_POR_NEURONIO = 500

# Ajustes específicos de uma arquitetura (sobrescrevem a base). Começa vazio.
# Exemplo de uso:  AJUSTES = {(16, 16): {"taxa_inicial": 0.3}}
AJUSTES = {}


def criar_som(linhas, colunas, **sobrescrever):
    """
    Cria UM SOM com os hiperparâmetros da configuração.
    `sobrescrever` permite testar um valor na hora, ex.: criar_som(8, 8, taxa_inicial=0.3)
    """
    params = {
        **CONFIG_BASE,
        "n_iteracoes": linhas * colunas * ITERACOES_POR_NEURONIO,
        **AJUSTES.get((linhas, colunas), {}),
        **sobrescrever,
    }
    return SOM(linhas, colunas, **params)


def criar_todos_som(**sobrescrever):
    """Cria os 3 SOMs do projeto (4x4, 8x8 e 16x16), sem treinar."""
    return [criar_som(l, c, **sobrescrever) for l, c in GRIDS]