from abc import ABC, abstractmethod

import numpy as np


class RedeBase(ABC):
    """Classe base abstrata para SOM e GNG."""

    # ------------------------------------------------------------------
    # PARTE 1: o que cada rede é obrigada a implementar
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def tipo(self):
        """'som' ou 'gng'. Usado para organizar pastas e nomes de arquivos."""

    @property
    @abstractmethod
    def arquitetura(self):
        """Texto curto da arquitetura: '8x8' (SOM) ou '64' (GNG)."""

    @abstractmethod
    def treinar(self, pixels):
        """
        Treina a rede com os pixels (N, 3) em [0, 1].
        Não devolve nada: o aprendizado fica guardado dentro da própria rede.
        """

    @abstractmethod
    def obter_pesos(self):
        """
        Devolve as cores aprendidas, array (K, 3) em [0, 1].
        K = número de neurônios (SOM 8x8 -> 64; GNG -> quantos existirem no fim).
        A linha i desse array é a cor do neurônio de índice i.
        """

    @abstractmethod
    def sao_vizinhos(self, i, j):
        """
        Diz se os neurônios de índice i e j são vizinhos na topologia da rede.

        Cada rede define "vizinho" do seu jeito:
            SOM : estão lado a lado no grid (incluindo diagonais)
            GNG : estão ligados por uma aresta
        O erro topológico precisa disso (ver `metrics`).
        """

    # ------------------------------------------------------------------
    # PARTE 2: o que a interface já entrega pronto
    # (funciona igual para SOM e GNG, pois só depende dos pesos)
    # ------------------------------------------------------------------

    def encontrar_bmus(self, pixels, tamanho_lote=4096):
        """
        Para cada pixel, acha o neurônio vencedor (BMU) e o segundo colocado.

        BMU = Best Matching Unit = neurônio cujo peso (cor) é o MAIS PARECIDO
        com o pixel, ou seja, o de menor distância euclidiana.

        Devolve (bmu1, bmu2), dois arrays (N,) de índices.

        Exemplo simples:
            pesos  = [[1,0,0], [0,0,1], [0.9,0.1,0]]   (vermelho, azul, quase vermelho)
            pixel  = [1,0,0]
            distâncias -> neurônio 0: 0.00 | neurônio 1: 1.41 | neurônio 2: 0.14
            bmu1 = 0 (vencedor), bmu2 = 2 (segundo colocado)

        Os pixels são processados em lotes para não gastar muita memória:
        sem isso, 1 milhão de pixels x 256 neurônios geraria uma matriz enorme.
        """
        pixels = np.asarray(pixels, dtype=np.float64)
        pesos = np.asarray(self.obter_pesos(), dtype=np.float64)

        if pesos.shape[0] < 2:
            raise ValueError("São necessários pelo menos 2 neurônios para achar o 2º colocado")

        n = len(pixels)
        bmu1 = np.empty(n, dtype=np.int64)
        bmu2 = np.empty(n, dtype=np.int64)

        for ini in range(0, n, tamanho_lote):
            lote = pixels[ini:ini + tamanho_lote]                       # (L, 3)
            # Distância ao quadrado de cada pixel do lote a cada neurônio -> (L, K)
            # (não precisa da raiz: a ordem dos menores é a mesma)
            dif = lote[:, None, :] - pesos[None, :, :]
            dist2 = np.sum(dif * dif, axis=2)
            # Os 2 menores de cada linha, já em ordem: coluna 0 = 1º, coluna 1 = 2º
            dois_menores = np.argpartition(dist2, 1, axis=1)[:, :2]
            d_dois = np.take_along_axis(dist2, dois_menores, axis=1)
            ordem = np.argsort(d_dois, axis=1)
            dois_menores = np.take_along_axis(dois_menores, ordem, axis=1)

            bmu1[ini:ini + len(lote)] = dois_menores[:, 0]
            bmu2[ini:ini + len(lote)] = dois_menores[:, 1]

        return bmu1, bmu2

    def quantizar(self, pixels):
        """
        Etapa de INFERÊNCIA do enunciado: índice do neurônio vencedor de cada pixel.

        O resultado é exatamente o `bmu_idx` que o `reconstruir_imagem`
        (image_utils) espera para montar a imagem quantizada.
        """
        bmu1, _ = self.encontrar_bmus(pixels)
        return bmu1

    # ------------------------------------------------------------------
    # PARTE 3: conveniência
    # ------------------------------------------------------------------

    @property
    def nome(self):
        """Nome curto da rede, ex.: 'som_8x8' ou 'gng_64'. Usado em arquivos e logs."""
        return f"{self.tipo}_{self.arquitetura}"

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.nome}>"