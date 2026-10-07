# Classificação de Imagens: Homer vs Bart (CNN + Transfer Learning)

Projeto de visão computacional que usa uma Rede Neural Convolucional (CNN) para
classificar imagens dos personagens **Homer** e **Bart**, dos Simpsons.
Desenvolvido como trabalho da faculdade (FIAP — Tecnólogo em IA), com foco em
entender cada etapa, não só em obter o resultado.

## Objetivo

Construir um classificador de imagens para dois personagens e, a partir dele,
comparar duas abordagens:

1. Uma **CNN treinada do zero**.
2. Um modelo com **Transfer Learning** (VGG16 pré-treinada), para avaliar o
   ganho de reaproveitar uma rede já treinada.

## Dataset

- **196 imagens de treino** (118 Bart / 78 Homer)
- **73 imagens de teste** (42 Bart / 31 Homer)
- Imagens no formato `.bmp`, organizadas em pastas por classe
  (`training_set/` e `test_set/`, cada uma com `bart/` e `homer/`).

> As imagens são de personagens de marca registrada e foram usadas apenas para
> fins de estudo. Por isso o dataset não está incluído neste repositório.

## Abordagem

1. **Pré-processamento** — normalização das imagens e *data augmentation*
   (giro, zoom, espelho) apenas no treino, usando `ImageDataGenerator`.
2. **CNN do zero** — camadas `Conv2D` + `BatchNormalization` + `MaxPooling`,
   seguidas de camadas densas e `Dropout`.
3. **Avaliação honesta** — além da acurácia, matriz de confusão e recall/F1 por
   classe (o dataset é desbalanceado, então só a acurácia enganaria).
4. **Transfer Learning** — VGG16 pré-treinada no ImageNet, com os pesos
   congelados e apenas uma "cabeça" nova treinada para decidir Homer vs Bart.
5. **Teste final** — previsão em imagens novas baixadas da internet.

## Resultados

| Modelo | Acurácia no teste |
|---|---|
| CNN do zero | ~42% |
| Transfer Learning (VGG16) | ~77% |

A CNN do zero sofreu com **overfitting** por causa da pouca quantidade de
imagens: acertava quase tudo no treino, mas no teste chutava "Homer" para quase
tudo (recall 0 para Bart). O **Transfer Learning** quase dobrou o acerto e
passou a classificar corretamente as duas classes, inclusive em imagens fora do
dataset.

## Principais aprendizados

- Com **poucos dados**, reaproveitar uma rede já treinada (Transfer Learning)
  funciona muito melhor do que treinar do zero.
- **Acurácia sozinha engana** em dataset desbalanceado — a matriz de confusão e
  o recall por classe mostram o que realmente está acontecendo.
- Cada modelo precisa do **pré-processamento compatível com o seu treino**
  (`rescale` na CNN do zero, `preprocess_input` na VGG16).

## Tecnologias

- Python
- TensorFlow / Keras
- VGG16 (Transfer Learning)
- scikit-learn (métricas)
- Matplotlib
- Google Colab

## Como rodar

1. Abra o notebook `CP05_CNN_para_Prever_Imagens.ipynb` no Google Colab.
2. Faça o upload do dataset (`dataset_personagens.zip`) quando a célula pedir.
3. Execute as células na ordem.

## Autor

Tiago Cesaro — Tecnólogo em Inteligência Artificial (FIAP)
