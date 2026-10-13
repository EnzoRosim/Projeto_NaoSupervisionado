"""
gng_configs.py
--------------
Arquiteturas e hiperparâmetros da GNG em UM só lugar.
Para ajustar algo (o enunciado pede "ajustes nos hiperparâmetros"), mexa aqui,
sem precisar tocar no modelo.

Arquiteturas pedidas no enunciado: redes com 16, 64 e 256 neurônios.
"""

from src.gng.gng_model import GNG

# Hiperparâmetros comuns a todas as arquiteturas (valores clássicos do artigo de Fritzke)
CONFIG_BASE = {
    "intervalo_insercao": 100,   # lambda: a cada 100 pixels, tenta inserir um neurônio
    "eps_vencedor": 0.2,
    "eps_vizinho": 0.006,
    "alpha": 0.5,
    "beta": 0.0005,
    "idade_maxima": 50,
    "semente": 42,
}

# Limites de neurônios do enunciado
NEURONIOS = [16, 64, 256]

# Quantas iterações de treino por neurônio (mesma regra do SOM, para comparar de forma justa).
# Exemplo: 16 neurônios -> 8.000 iterações | 256 neurônios -> 128.000 iterações.
# Lembrete: a rede só cresce 1 neurônio a cada `intervalo_insercao` iterações. Chegar
# a 256 neurônios leva ~ 254 * 100 = 25.400 iterações; o resto do treino é refinamento.
ITERACOES_POR_NEURONIO = 500

# Ajustes específicos de uma arquitetura (sobrescrevem a base). Começa vazio.
# Exemplo de uso:  AJUSTES = {256: {"intervalo_insercao": 50}}
AJUSTES = {}


def criar_gng(max_neuronios, **sobrescrever):
    """
    Cria UMA GNG com os hiperparâmetros da configuração.
    `sobrescrever` permite testar um valor na hora, ex.: criar_gng(64, eps_vencedor=0.1)
    """
    params = {
        **CONFIG_BASE,
        "n_iteracoes": max_neuronios * ITERACOES_POR_NEURONIO,
        **AJUSTES.get(max_neuronios, {}),
        **sobrescrever,
    }
    return GNG(max_neuronios, **params)


def criar_todas_gng(**sobrescrever):
    """Cria as 3 GNGs do projeto (16, 64 e 256 neurônios), sem treinar."""
    return [criar_gng(n, **sobrescrever) for n in NEURONIOS]