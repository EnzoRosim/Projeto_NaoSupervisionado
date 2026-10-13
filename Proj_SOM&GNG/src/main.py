import argparse
import sys
from pathlib import Path

# main.py fica em <raiz>/src/main.py, então a raiz do projeto é a pasta acima de src/
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))   # permite `from src...` mesmo rodando de outra pasta

import matplotlib  # noqa: E402
matplotlib.use("Agg")           # só salva as figuras em arquivo, sem abrir janelas

from src.experiments.run_experiments import executar  # noqa: E402

LADO_MAXIMO_PADRAO = 128        # maior lado das imagens, em pixels (igual para todas)


def ler_argumentos():
    p = argparse.ArgumentParser(description="Quantização de cores com SOM e GNG")
    p.add_argument("--imagens", nargs="+", metavar="NOME",
                   help="nomes das imagens (sem extensão); padrão: todas de database/")
    p.add_argument("--tipo", choices=["som", "gng", "todos"], default="todos",
                   help="quais redes rodar (padrão: todos)")
    p.add_argument("--lado-maximo", type=int, default=LADO_MAXIMO_PADRAO, metavar="PX",
                   help=f"maior lado das imagens em pixels (padrão: {LADO_MAXIMO_PADRAO})")
    return p.parse_args()


def main():
    args = ler_argumentos()
    tipos = ("som", "gng") if args.tipo == "todos" else (args.tipo,)

    try:
        executar(RAIZ, nomes_imagens=args.imagens, lado_maximo=args.lado_maximo, tipos=tipos)
    except FileNotFoundError as erro:
        print(f"Erro: {erro}")
        sys.exit(1)

    print(f"\nLog:      {RAIZ / 'output' / 'logs' / 'ultima_run.log'}")
    print(f"Gráficos: {RAIZ / 'output' / 'graphics'}")


if __name__ == "__main__":
    main()