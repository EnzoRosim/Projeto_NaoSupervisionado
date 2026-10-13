"""
gng_model.py
------------
Growing Neural Gas (GNG, Fritzke 1995) para quantização de cores.

Diferença para o SOM, em uma frase: no SOM o grid é FIXO e definido antes do
treino. Na GNG a rede COMEÇA com 2 neurônios e CRESCE sozinha, inserindo
neurônios onde o erro é maior, até chegar ao limite pedido (16, 64 ou 256).
A "vizinhança" não é um grid: são ARESTAS entre neurônios, criadas e removidas
durante o treino.

Algoritmo (a cada pixel sorteado x):
    1. Acha o vencedor s1 e o segundo colocado s2 (os 2 neurônios mais parecidos com x).
    2. Soma ao erro de s1 a distância² até x (s1 acumula "o quanto errou").
    3. Move s1 em direção a x (passo grande, eps_b) e seus vizinhos
       ligados por aresta também (passo pequeno, eps_n).
    4. Envelhece as arestas de s1. Cria (ou rejuvenesce, idade=0) a aresta s1-s2.
    5. Remove arestas velhas demais (idade > a_max) e neurônios que ficaram sem arestas.
    6. A cada `intervalo_insercao` pixels, se ainda há espaço: insere um neurônio novo
       NO MEIO entre o neurônio de maior erro (q) e o vizinho dele de maior erro (f).
    7. Todos os erros diminuem um pouco (beta), para dar mais peso ao erro recente.

Exemplo mental (cores em 1D, só para visualizar):
    começa:    [ azul ]---[ amarelo ]
    o azul acumula muito erro (há pixels verdes e roxos que ele representa mal)
    insere:    [ azul ]---[ NOVO ]---[ amarelo ]    NOVO = cor intermediária (azul+amarelo)/2
    o treino então empurra o NOVO para onde há pixels de fato.
"""

import numpy as np

from src.common.base_network import RedeBase


class GNG(RedeBase):

    def __init__(
        self,
        max_neuronios,
        n_iteracoes=20000,
        intervalo_insercao=100,
        eps_vencedor=0.2,
        eps_vizinho=0.006,
        alpha=0.5,
        beta=0.0005,
        idade_maxima=50,
        semente=42,
    ):
        """
        max_neuronios      : limite de crescimento (a "arquitetura": 16, 64 ou 256)
        n_iteracoes        : quantos pixels sorteados serão usados no treino
        intervalo_insercao : a cada quantos pixels tenta inserir um neurônio novo (lambda)
        eps_vencedor       : o quanto o vencedor anda em direção ao pixel (0 a 1)
        eps_vizinho        : o quanto os vizinhos do vencedor andam (bem menor)
        alpha              : fator que reduz o erro de q e f quando nasce um neurônio entre eles
        beta               : decaimento do erro de todos os neurônios a cada iteração
        idade_maxima       : aresta mais velha que isso é removida (a_max)
        semente            : fixa o sorteio, para os resultados serem reproduzíveis
        """
        if max_neuronios < 2:
            raise ValueError("max_neuronios deve ser pelo menos 2")
        if not (0 < eps_vizinho < eps_vencedor <= 1):
            raise ValueError("Use 0 < eps_vizinho < eps_vencedor <= 1")

        self.max_neuronios = max_neuronios
        self.n_iteracoes = n_iteracoes
        self.intervalo_insercao = intervalo_insercao
        self.eps_vencedor = eps_vencedor
        self.eps_vizinho = eps_vizinho
        self.alpha = alpha
        self.beta = beta
        self.idade_maxima = idade_maxima
        self.semente = semente

        self._pesos = None        # (K, 3) final, só com os neurônios que sobraram
        self._adjacencia = None   # (K, K) bool final: True se há aresta entre i e j

    # ------------------------------------------------------------------
    # Contrato da RedeBase
    # ------------------------------------------------------------------

    @property
    def tipo(self):
        return "gng"

    @property
    def arquitetura(self):
        # O limite pedido (16, 64, 256). O número REAL ao fim do treino fica em `n_neuronios`.
        return str(self.max_neuronios)

    def obter_pesos(self):
        if self._pesos is None:
            raise RuntimeError("A rede ainda não foi treinada: chame treinar(pixels) primeiro")
        return self._pesos

    def sao_vizinhos(self, i, j):
        """
        Dois neurônios são vizinhos na GNG se existe uma ARESTA entre eles.
        (Os índices i, j são os da matriz devolvida por `obter_pesos`.)
        """
        if self._adjacencia is None:
            raise RuntimeError("A rede ainda não foi treinada: chame treinar(pixels) primeiro")
        return bool(i != j and self._adjacencia[i, j])

    # ------------------------------------------------------------------
    # Informações úteis depois do treino
    # ------------------------------------------------------------------

    @property
    def n_neuronios(self):
        """Quantos neurônios a rede tem de fato ao fim do treino."""
        return len(self.obter_pesos())

    @property
    def n_arestas(self):
        """Quantas arestas (conexões) a rede tem ao fim do treino."""
        self.obter_pesos()
        return int(self._adjacencia.sum() // 2)

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------

    def treinar(self, pixels, verbose=False):
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 3:
            raise ValueError(f"`pixels` deve ter forma (N, 3), mas veio {pixels.shape}")
        if len(pixels) < 2:
            raise ValueError("São necessários pelo menos 2 pixels")

        rng = np.random.default_rng(self.semente)
        m = self.max_neuronios

        # Estruturas com tamanho fixo M (o máximo de neurônios). Um "slot" livre
        # é um espaço onde pode nascer um neurônio novo.
        pesos = np.zeros((m, 3))
        erro = np.zeros(m)
        ativo = np.zeros(m, dtype=bool)
        # idade[i, j] = idade da aresta i-j, ou -1 se NÃO existe aresta.
        idade = np.full((m, m), -1, dtype=np.int64)

        # Começa com 2 neurônios, cada um com a cor de um pixel real sorteado
        inicio = rng.choice(len(pixels), size=2, replace=False)
        pesos[:2] = pixels[inicio]
        ativo[:2] = True

        sorteados = rng.integers(0, len(pixels), size=self.n_iteracoes)

        for t, idx in enumerate(sorteados):
            x = pixels[idx]

            # 1) Vencedor s1 e segundo colocado s2 (só entre neurônios ativos)
            d2 = np.sum((pesos - x) ** 2, axis=1)
            d2[~ativo] = np.inf
            dois = np.argpartition(d2, 1)[:2]
            s1, s2 = (dois[0], dois[1]) if d2[dois[0]] <= d2[dois[1]] else (dois[1], dois[0])

            # 2) Acumula o erro do vencedor
            erro[s1] += d2[s1]

            # 3) Move o vencedor e seus vizinhos em direção ao pixel
            pesos[s1] += self.eps_vencedor * (x - pesos[s1])
            vizinhos = np.flatnonzero(idade[s1] >= 0)
            if len(vizinhos):
                pesos[vizinhos] += self.eps_vizinho * (x - pesos[vizinhos])

            # 4) Envelhece as arestas de s1; cria/rejuvenesce a aresta s1-s2
            idade[s1, vizinhos] += 1
            idade[vizinhos, s1] += 1
            idade[s1, s2] = 0
            idade[s2, s1] = 0

            # 5) Remove arestas velhas e neurônios que ficaram sem nenhuma aresta
            velhos = np.flatnonzero(idade[s1] > self.idade_maxima)
            if len(velhos):
                idade[s1, velhos] = -1
                idade[velhos, s1] = -1
                for n in velhos:
                    if not (idade[n] >= 0).any():      # ficou isolado
                        ativo[n] = False
                        erro[n] = 0.0

            # 6) A cada `intervalo_insercao` pixels, insere um neurônio novo (se houver espaço)
            if (t + 1) % self.intervalo_insercao == 0 and not ativo.all():
                self._inserir_neuronio(pesos, erro, ativo, idade)

            # 7) Todos os erros decaem um pouco
            erro *= (1.0 - self.beta)

            if verbose and (t + 1) % max(self.n_iteracoes // 5, 1) == 0:
                print(f"  [{self.nome}] {t + 1}/{self.n_iteracoes} iterações, "
                      f"{int(ativo.sum())} neurônios")

        # Compacta: guarda só os neurônios ativos, na ordem dos slots
        ativos = np.flatnonzero(ativo)
        self._pesos = np.clip(pesos[ativos], 0.0, 1.0)
        self._adjacencia = idade[np.ix_(ativos, ativos)] >= 0
        return self

    def _inserir_neuronio(self, pesos, erro, ativo, idade):
        """
        Insere um neurônio r entre q (maior erro) e f (vizinho de q com maior erro).

        Exemplo:  antes   q ---------- f
                  depois  q --- r --- f      (r = cor média de q e f)
        A aresta q-f é trocada por q-r e r-f, e o erro de q e f cai pela metade
        (alpha), pois r vai "dividir o trabalho" com eles.
        """
        erro_ativos = np.where(ativo, erro, -np.inf)
        q = int(np.argmax(erro_ativos))

        vizinhos_q = np.flatnonzero(idade[q] >= 0)
        if len(vizinhos_q) == 0:
            return                                           # q isolado: nada a fazer
        f = int(vizinhos_q[np.argmax(erro[vizinhos_q])])

        r = int(np.flatnonzero(~ativo)[0])                   # primeiro slot livre
        pesos[r] = 0.5 * (pesos[q] + pesos[f])

        idade[q, f] = idade[f, q] = -1
        idade[q, r] = idade[r, q] = 0
        idade[r, f] = idade[f, r] = 0

        erro[q] *= self.alpha
        erro[f] *= self.alpha
        erro[r] = erro[q]
        ativo[r] = True