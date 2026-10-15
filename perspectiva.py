import glob
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

PASTA_FOTOS = "fotos"
PASTA_SAIDA = "resultados"


def detectar_bordas(imagem, kernel_desfoque=(5, 5), canny_minimo=50, canny_maximo=150):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, kernel_desfoque, 0)
    return cv2.Canny(cinza, canny_minimo, canny_maximo)


def encontrar_quadrilatero(bordas, max_candidatos=5, eps_aproximacao=0.02):
    contornos, _ = cv2.findContours(bordas.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:max_candidatos]

    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        aprox = cv2.approxPolyDP(contorno, eps_aproximacao * perimetro, True)
        if len(aprox) == 4:
            return aprox.reshape(4, 2)

    raise ValueError("Nenhum contorno foi encontrado")


def ordenar_cantos(cantos):
    cantos = sorted(cantos, key=lambda ponto: ponto[0])
    esquerda = sorted(cantos[:2], key=lambda ponto: ponto[1])
    direita = sorted(cantos[2:], key=lambda ponto: ponto[1])
    superior_esquerdo, inferior_esquerdo = esquerda
    superior_direito, inferior_direito = direita
    return np.array(
        [superior_esquerdo, superior_direito, inferior_direito, inferior_esquerdo],
        dtype=np.float32,
    )


def aplicar_transformacao(imagem, cantos, largura_saida, altura_saida):
    destino = np.array(
        [[0, 0], [largura_saida, 0], [largura_saida, altura_saida], [0, altura_saida]],
        dtype=np.float32,
    )
    matriz_transformacao = cv2.getPerspectiveTransform(cantos, destino)
    return cv2.warpPerspective(imagem, matriz_transformacao, (largura_saida, altura_saida))


def salvar_figura(imagem, titulo, caminho_saida, mapa_cores=None):
    plt.figure()
    plt.imshow(imagem if mapa_cores else cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB), cmap=mapa_cores)
    plt.title(titulo)
    plt.axis("off")
    plt.savefig(caminho_saida, bbox_inches="tight")
    plt.close()


def transformar_perspectiva(caminho_imagem, largura_saida=800, altura_saida=1000, pasta_saida=PASTA_SAIDA):
    nome = os.path.splitext(os.path.basename(caminho_imagem))[0]

    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise FileNotFoundError(f"nao foi possível carregar a imagem {caminho_imagem}")

    imagem = cv2.resize(imagem, (largura_saida, altura_saida))
    original = imagem.copy()

    bordas = detectar_bordas(imagem)
    salvar_figura(
        bordas, "Pré-processamento",
        os.path.join(pasta_saida, "preprocessamento", f"{nome}.jpg"), mapa_cores="gray",
    )

    cantos = encontrar_quadrilatero(bordas)

    pre_visualizacao = original.copy()
    cv2.drawContours(pre_visualizacao, [cantos.astype(np.int32)], -1, (0, 0, 255), 3)
    salvar_figura(
        pre_visualizacao, "Detecção de cantos",
        os.path.join(pasta_saida, "deteccao", f"{nome}.jpg"),
    )

    cantos_ordenados = ordenar_cantos(cantos)
    transformada = aplicar_transformacao(original, cantos_ordenados, largura_saida, altura_saida)

    fig, eixos = plt.subplots(1, 2, figsize=(12, 6))
    eixos[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    eixos[0].set_title("Imagem original")
    eixos[0].axis("off")
    eixos[1].imshow(cv2.cvtColor(transformada, cv2.COLOR_BGR2RGB))
    eixos[1].set_title("Imagem transformada")
    eixos[1].axis("off")
    fig.savefig(os.path.join(pasta_saida, "transformacao", f"{nome}.jpg"), bbox_inches="tight")
    plt.close(fig)

    return transformada


def processar_pasta(pasta=PASTA_FOTOS, pasta_saida=PASTA_SAIDA, largura_saida=800, altura_saida=1000):
    for sub in ("preprocessamento", "deteccao", "transformacao"):
        os.makedirs(os.path.join(pasta_saida, sub), exist_ok=True)

    caminhos = sorted(
        glob.glob(os.path.join(pasta, "*.jpg"))
        + glob.glob(os.path.join(pasta, "*.jpeg"))
        + glob.glob(os.path.join(pasta, "*.png"))
    )
    if not caminhos:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em '{pasta}'")

    resultados = []
    for caminho in caminhos:
        nome = os.path.basename(caminho)
        try:
            resultados.append(transformar_perspectiva(caminho, largura_saida, altura_saida, pasta_saida))
            print(f"{nome}: ok")
        except ValueError as e:
            print(f"{nome}: {e} -- pulando")
    return resultados


if __name__ == "__main__":
    processar_pasta()
