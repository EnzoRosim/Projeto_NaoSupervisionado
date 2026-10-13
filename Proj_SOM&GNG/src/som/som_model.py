"""
som_model.py
------------
Self-Organizing Map (SOM, mapa de Kohonen) para quantização de cores.

Ideia em uma frase: os neurônios ficam num grid (ex.: 8x8) e cada um guarda
uma cor (seus "pesos" R, G, B). Para cada pixel de treino, o neurônio cuja cor
mais se parece com o pixel (o BMU) "ganha" e se aproxima do pixel. Seus vizinhos
NO GRID também se aproximam, só que menos. Com o tempo, cores parecidas acabam
em neurônios vizinhos (preservação topológica).

Exemplo mental (grid 1D, 5 neurônios, pixel azul):
    [ vermelho | laranja | AZUL-ESCURO | verde | amarelo ]
                            ^ BMU
    - o BMU anda bastante em direção ao azul
    - os vizinhos imediatos (laranja e verde) andam um pouco
    - os mais distantes quase não se mexem

Dois hiperparâmetros decaem durante o treino (começa explorando, termina refinando):
    - taxa de aprendizado : o quanto o BMU anda em direção ao pixel
    - raio de vizinhança  : até onde o "puxão" alcança no grid
"""

import numpy as np

from src.common.base_network import RedeBase


class SOM(RedeBase):

    def __init__(
        self,
        linhas,
        colunas,
        n_iteracoes=20000,
        taxa_inicial=0.5,
        taxa_final=0.01,
        raio_inicial=None,
        raio_final=0.5,
        semente=42,
    ):
        """
        linhas, colunas : tamanho do grid (SOM 8x8 -> linhas=8, colunas=8)
        n_iteracoes     : quantos pixels sorteados serão usados no treino
        taxa_inicial/final : taxa de aprendizado no começo e no fim do treino
        raio_inicial/final : raio de vizinhança (em "casas" do grid) no começo e no fim
        raio_inicial=None  : usa metade do maior lado do grid
        semente         : fixa o sorteio, para os resultados serem reproduzíveis
        """
        if linhas < 1 or colunas < 1 or linhas * colunas < 2:
            raise ValueError("O grid precisa ter pelo menos 2 neurônios")
        if not (0 < taxa_final <= taxa_inicial):
            raise ValueError("Use 0 < taxa_final <= taxa_inicial")

        self.linhas = linhas
        self.colunas = colunas
        self.n_iteracoes = n_iteracoes
        self.taxa_inicial = taxa_inicial
        self.taxa_final = taxa_final
        self.raio_inicial = raio_inicial if raio_inicial is not None else max(linhas, colunas) / 2
        self.raio_final = raio_final
        if not (0 < self.raio_final <= self.raio_inicial):
            raise ValueError("Use 0 < raio_final <= raio_inicial")
        self.semente = semente

        # Posição de cada neurônio no grid. O neurônio de índice i fica em
        # (linha, coluna) = (i // colunas, i % colunas).
        # Exemplo grid 2x3:  índice 0 1 2 / 3 4 5  ->  posições (0,0) (0,1) (0,2) / (1,0) (1,1) (1,2)
        indices = np.arange(linhas * colunas)
        self._posicoes = np.stack([indices // colunas, indices % colunas], axis=1).astype(np.float64)

        self._pesos = None   # (K, 3); só existe depois de treinar

    # ------------------------------------------------------------------
    # Contrato da RedeBase
    # ------------------------------------------------------------------

    @property
    def tipo(self):
        return "som"

    @property
    def arquitetura(self):
        return f"{self.linhas}x{self.colunas}"

    def obter_pesos(self):
        if self._pesos is None:
            raise RuntimeError("A rede ainda não foi treinada: chame treinar(pixels) primeiro")
        return self._pesos

    def sao_vizinhos(self, i, j):
        """
        Dois neurônios são vizinhos se estão lado a lado no grid, diagonais
        incluídas (no máximo 1 casa de diferença em linha e em coluna).

        Exemplo (grid 8x8): o neurônio na posição (3,3) é vizinho de (2,2), (3,4), (4,3)...
        mas NÃO de (3,5) (2 casas de distância).
        """
        if i == j:
            return False
        dif = np.abs(self._posicoes[i] - self._posicoes[j])
        return bool(dif.max() <= 1)

    def treinar(self, pixels, verbose=False):
        """
        Treina o SOM. A cada iteração t (de 0 a n_iteracoes-1):
            1. sorteia um pixel
            2. acha o BMU (neurônio mais parecido)
            3. calcula taxa(t) e raio(t), que diminuem com o tempo
            4. puxa BMU e vizinhos em direção ao pixel, com força que cai com a
               distância no grid (função gaussiana)
        """
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 3:
            raise ValueError(f"`pixels` deve ter forma (N, 3), mas veio {pixels.shape}")
        if len(pixels) == 0:
            raise ValueError("`pixels` está vazio")

        rng = np.random.default_rng(self.semente)
        k = self.linhas * self.colunas

        # Inicialização: cada neurônio começa com a cor de um pixel real sorteado.
        # (Começar de pixels reais evita neurônios "mortos" em cores que nem existem na imagem.)
        inicio = rng.choice(len(pixels), size=k, replace=len(pixels) < k)
        pesos = pixels[inicio].copy()

        sorteados = rng.integers(0, len(pixels), size=self.n_iteracoes)

        for t, idx in enumerate(sorteados):
            fracao = t / max(self.n_iteracoes - 1, 1)             # 0 no início, 1 no fim
            taxa = self._decair(self.taxa_inicial, self.taxa_final, fracao)
            raio = self._decair(self.raio_inicial, self.raio_final, fracao)

            x = pixels[idx]

            # 2) BMU: menor distância (ao quadrado) entre o pixel e a cor de cada neurônio
            bmu = np.argmin(np.sum((pesos - x) ** 2, axis=1))

            # 4) Influência de cada neurônio: gaussiana da distância NO GRID ao BMU
            dist2_grid = np.sum((self._posicoes - self._posicoes[bmu]) ** 2, axis=1)
            influencia = np.exp(-dist2_grid / (2.0 * raio ** 2))           # (K,)

            # Cada peso anda em direção ao pixel: novo = antigo + taxa * influência * (pixel - antigo)
            pesos += (taxa * influencia)[:, None] * (x - pesos)

            if verbose and (t + 1) % max(self.n_iteracoes // 5, 1) == 0:
                print(f"  [{self.nome}] {t + 1}/{self.n_iteracoes} iterações  (taxa={taxa:.3f}, raio={raio:.2f})")

        self._pesos = np.clip(pesos, 0.0, 1.0)
        return self

    # ------------------------------------------------------------------
    # Auxiliar
    # ------------------------------------------------------------------

    @staticmethod
    def _decair(valor_inicial, valor_final, fracao):
        """
        Decaimento exponencial de valor_inicial até valor_final.

        Exemplo: de 0.5 até 0.01.
            fracao = 0.0 -> 0.5     (começo)
            fracao = 0.5 -> 0.0707  (meio: a média geométrica, não a aritmética)
            fracao = 1.0 -> 0.01    (fim)
        Cai rápido no início e devagar no fim, o que ajuda o SOM a se organizar
        primeiro e refinar depois.
        """
        return valor_inicial * (valor_final / valor_inicial) ** fracao