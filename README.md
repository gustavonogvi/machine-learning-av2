# Vehicle Collision Data in Seattle (2005–2019)

Projeto de Aprendizado de Máquina — T326 Ciência de Dados | Universidade de Fortaleza

## Sobre o projeto

Análise e modelagem preditiva sobre dados de colisões veiculares em Seattle entre 2005 e 2019. O objetivo é classificar a severidade dos acidentes com base em condições climáticas, de trânsito e comportamentais.

## Dataset

O dataset **não está incluído** no repositório. Faça o download manualmente:

1. Acesse: https://www.kaggle.com/datasets/mcfly1/vehicle-collision-data-in-seattle-2005-2019
2. Baixe o arquivo `seattle_collision_data_2005_2019.csv`
3. Coloque o arquivo na pasta `data/`:

```
machine-learning-av2/
└── data/
    └── seattle_collision_data_2005_2019.csv
```

Ou use o kagglehub (requer conta no Kaggle e API key configurada):

```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('mcfly1/vehicle-collision-data-in-seattle-2005-2019')"
```

## Requisitos

- Python 3.10+
- Instale as dependências:

```bash
pip install -r requirements.txt
```

## Como rodar

Execute os notebooks na ordem:

```
01_eda.ipynb           → Análise exploratória dos dados
02_preprocessing.ipynb → Pré-processamento e feature engineering
03_modeling.ipynb      → Treinamento e avaliação dos modelos
```

## Estrutura do projeto

```
machine-learning-av2/
├── data/                  # Dataset (não versionado)
├── 01_eda.ipynb
├── 02_preprocessing.ipynb
├── 03_modeling.ipynb
├── requirements.txt
└── README.md
```
