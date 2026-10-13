import csv
import time
from datetime import datetime
from pathlib import Path

from src.common.image_utils import (
    carregar_imagem, imagem_para_pixels, reconstruir_imagem, paleta_unica, listar_imagens,
)
from src.common.metrics import avaliar_rede
from src.common.plots import plotar_comparacao, plotar_metricas_imagem
from src.gng.gng_configs import criar_todas_gng
from src.som.som_configs import criar_todos_som

# Cada tipo de rede sabe criar a sua lista de 3 arquiteturas (definidas nos *_configs)
CRIADORES = {"som": criar_todos_som, "gng": criar_todas_gng}

CAMPOS_CSV = ["imagem", "tipo", "arquitetura", "neuronios",
              "erro_quantizacao", "erro_topologico", "tempo_s", "parametros"]


def executar(raiz, nomes_imagens=None, lado_maximo=128, tipos=("som", "gng")):
    raiz = Path(raiz)
    imagens = _escolher_imagens(raiz / "database", nomes_imagens)
    pasta_graficos = raiz / "output" / "graphics"
    pasta_logs = raiz / "output" / "logs"
    pasta_logs.mkdir(parents=True, exist_ok=True)

    n_redes = sum(len(CRIADORES[t]()) for t in tipos)
    total = len(imagens) * n_redes
    resultados = []
    contador = 0
    inicio_geral = time.time()

    # utf-8-sig no CSV: o Excel abre os acentos corretamente
    with open(pasta_logs / "ultima_run.log", "w", encoding="utf-8") as log, \
         open(pasta_logs / "ultima_run.csv", "w", newline="", encoding="utf-8-sig") as arq_csv:

        escritor = csv.DictWriter(arq_csv, fieldnames=CAMPOS_CSV, delimiter=";")
        escritor.writeheader()

        _emitir(log, "=" * 100)
        _emitir(log, f"ÚLTIMA RUN - {datetime.now():%d/%m/%Y %H:%M:%S}")
        _emitir(log, f"Imagens: {', '.join(p.stem for p in imagens)}")
        _emitir(log, f"Redes: {', '.join(t.upper() for t in tipos)} | "
                     f"lado máximo das imagens: {lado_maximo}px | total: {total} redes")
        _emitir(log, "=" * 100)

        for caminho in imagens:
            nome_img = caminho.stem

            # 1) Extração
            imagem = carregar_imagem(caminho, lado_maximo=lado_maximo)
            pixels = imagem_para_pixels(imagem)
            _emitir(log, f"\n>>> {nome_img}: {imagem.shape[1]}x{imagem.shape[0]} px, "
                         f"{len(pixels)} pixels, {paleta_unica(imagem)} cores distintas")

            resultados_img = []
            for tipo in tipos:
                for rede in CRIADORES[tipo]():       # redes novas (sem treino) para cada imagem
                    contador += 1

                    # 2) Treino
                    t0 = time.time()
                    rede.treinar(pixels)
                    segundos = time.time() - t0

                    # 3) Avaliação
                    metricas = avaliar_rede(rede, pixels)

                    # 4) Inferência + reconstrução + figura
                    reconstruida = reconstruir_imagem(
                        rede.quantizar(pixels), rede.obter_pesos(), imagem.shape)
                    n_neuronios = len(rede.obter_pesos())
                    plotar_comparacao(
                        imagem, reconstruida, rede.obter_pesos(),
                        pasta_graficos / rede.tipo / nome_img / f"{nome_img}_{rede.nome}.png",
                        titulo=f"{nome_img} - {rede.tipo.upper()} {rede.arquitetura} "
                               f"({n_neuronios} neurônios)",
                    )

                    # 5) Registro
                    linha = {
                        "imagem": nome_img,
                        "tipo": rede.tipo,
                        "arquitetura": rede.arquitetura,
                        "neuronios": n_neuronios,
                        "erro_quantizacao": round(metricas["erro_quantizacao"], 6),
                        "erro_topologico": round(metricas["erro_topologico"], 6),
                        "tempo_s": round(segundos, 2),
                        "parametros": _texto_parametros(rede),
                    }
                    resultados.append(linha)
                    resultados_img.append(linha)
                    escritor.writerow(linha)
                    arq_csv.flush()
                    _emitir(log, f"[{contador:>2}/{total}] {rede.tipo.upper()} {rede.arquitetura:>5} | "
                                 f"neurônios={n_neuronios:>3} | "
                                 f"EQ={linha['erro_quantizacao']:.4f} | ET={linha['erro_topologico']:.4f} | "
                                 f"{segundos:5.1f}s")
                    log.write(f"        parâmetros: {linha['parametros']}\n")
                    log.flush()

            # Gráfico comparando as redes desta imagem
            plotar_metricas_imagem(
                resultados_img, pasta_graficos / "comparativo" / f"{nome_img}_metricas.png",
                titulo=f"{nome_img} - métricas das redes")

        _escrever_resumo(log, resultados)
        total_s = time.time() - inicio_geral
        _emitir(log, f"\nConcluído em {total_s:.0f}s ({total_s / 60:.1f} min).")

    return resultados


# ----------------------------------------------------------------------
# Auxiliares internas
# ----------------------------------------------------------------------

def _escolher_imagens(pasta_database, nomes):
    """Todas as imagens de database/ ou só as pedidas (pelo nome sem extensão)."""
    todas = listar_imagens(pasta_database)
    if not todas:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {pasta_database}")
    if not nomes:
        return todas

    por_nome = {p.stem.lower(): p for p in todas}
    escolhidas = []
    for nome in nomes:
        chave = Path(nome).stem.lower()
        if chave not in por_nome:
            disponiveis = ", ".join(p.stem for p in todas)
            raise FileNotFoundError(f"Imagem '{nome}' não encontrada. Disponíveis: {disponiveis}")
        escolhidas.append(por_nome[chave])
    return escolhidas


def _texto_parametros(rede):
    itens = {k: v for k, v in vars(rede).items()
             if not k.startswith("_") and isinstance(v, (int, float))}
    return " ".join(f"{k}={v}" for k, v in itens.items())


def _escrever_resumo(log, resultados):
    """Média por arquitetura sobre todas as imagens (útil para o relatório)."""
    grupos = {}
    for r in resultados:
        grupos.setdefault((r["tipo"], r["arquitetura"]), []).append(r)

    _emitir(log, "\n" + "=" * 100)
    _emitir(log, f"RESUMO: média por rede sobre {len(set(r['imagem'] for r in resultados))} imagem(ns)")
    _emitir(log, "=" * 100)
    _emitir(log, f"{'rede':<12}{'EQ médio':>12}{'ET médio':>12}{'tempo médio':>14}")
    for (tipo, arq), linhas in grupos.items():
        n = len(linhas)
        eq = sum(l["erro_quantizacao"] for l in linhas) / n
        et = sum(l["erro_topologico"] for l in linhas) / n
        seg = sum(l["tempo_s"] for l in linhas) / n
        _emitir(log, f"{tipo.upper() + ' ' + arq:<12}{eq:>12.4f}{et:>12.4f}{seg:>13.1f}s")


def _emitir(log, texto):
    """Mostra no terminal E grava no arquivo de log."""
    print(texto)
    log.write(texto + "\n")
    log.flush()