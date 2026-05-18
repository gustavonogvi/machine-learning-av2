# CLAUDE.md — machine-learning-av2

## Contexto do Projeto

Trabalho 2 da disciplina T326 - Ciência de Dados (Prof. Rilder de Sousa Pires, LCDIA - Universidade de Fortaleza).
Projeto completo de Machine Learning usando o dataset **Vehicle Collision Data in Seattle (2005–2019)**.
Tarefa: **classificação** da severidade dos acidentes (`SEVERITYCODE`).

## Dataset

- Arquivo: `data/seattle_collision_data_2005_2019.csv`
- 111.882 registros, 34 colunas
- Target: `SEVERITYCODE` (inteiro — severidade da colisão)
- Não versionar a pasta `data/` no git

## Regras de Trabalho

- Trabalhar **em blocos metodicos**, célula por célula, sem pular etapas
- Garantir **rigor matemático/estatístico** em cada etapa — comentar o porquê de cada decisão
- Não pular para modelagem sem pré-processamento completo
- Evitar **data leakage** — qualquer transformação que use estatísticas do conjunto inteiro deve ser feita após o split
- Usar **pelo menos 2 modelos** de ML
- Commits sem co-author

## Plano de Notebooks

### Bloco 1 — `01_eda.ipynb` (Análise Exploratória)
1. Carregamento e inspeção geral (shape, dtypes, nulos)
2. Análise da variável alvo `SEVERITYCODE` (distribuição de classes)
3. Análise univariada — numéricas (histogramas) e categóricas (barplots)
4. Análise bivariada — relação das features com o target
5. Matriz de correlação (heatmap)
6. Conclusões parciais da EDA

### Bloco 2 — `02_preprocessing.ipynb` (Pré-processamento)
1. Remoção de colunas irrelevantes e com risco de leakage
2. Tratamento de valores nulos
3. Tratamento de outliers
4. Codificação de variáveis categóricas (Label Encoding / One-Hot)
5. Normalização/padronização das features numéricas
6. Split treino/teste estratificado

### Bloco 3 — `03_modeling.ipynb` (Modelagem e Avaliação)
1. Modelo 1: Regressão Logística (baseline interpretável)
2. Modelo 2: Random Forest (ensemble, captura não-linearidade)
3. Métricas: accuracy, precision, recall, F1-score, matriz de confusão
4. Comparação entre modelos
5. Discussão de overfitting, limitações e possíveis melhorias

## Critérios de Avaliação (referência)

- **40% Conceitual:** qualidade das análises, domínio do conteúdo, interpretação dos resultados
- **30% Procedimental:** uso correto das técnicas, qualidade das visualizações
- **30% Atitudinal:** organização do GitHub/relatório, prazo, trabalho em equipe

## Restrições do Relatório

- Máximo 15 páginas
- Máximo 5 figuras
