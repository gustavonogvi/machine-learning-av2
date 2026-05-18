# CLAUDE.md — machine-learning-av2

## Contexto do Projeto

Trabalho 2 da disciplina T326 - Ciência de Dados (Prof. Rilder de Sousa Pires, LCDIA - Universidade de Fortaleza).
Projeto completo de Machine Learning usando o dataset **Vehicle Collision Data in Seattle (2005–2019)**.
Tarefa: **classificação** da severidade dos acidentes (`SEVERITYCODE`).

## Dataset

- Arquivo: `data/seattle_collision_data_2005_2019.csv`
- 111.882 registros, 34 colunas
- Target: `SEVERITYCODE` (inteiro — severidade da colisão, 4 classes: 0, 1, 2, 3)
- Não versionar a pasta `data/` no git

## Regras de Trabalho

- Trabalhar **em blocos metódicos**, célula por célula, sem pular etapas
- Garantir **rigor matemático/estatístico** em cada etapa — comentar o porquê de cada decisão
- Não pular para modelagem sem pré-processamento completo
- Evitar **data leakage** — qualquer transformação que use estatísticas do conjunto inteiro deve ser feita após o split
- Usar **pelo menos 2 modelos** de ML
- Commits sem co-author
- **Sem linhas de comentário no código** — explicações ficam nas células markdown do notebook

---

## Bloco 1 — `01_eda.ipynb` ✅ CONCLUÍDO

### Estrutura
1. Carregamento e inspeção geral (shape, dtypes, nulos)
2. Análise da variável alvo `SEVERITYCODE` (distribuição de classes + baseline)
3. Análise univariada — numéricas (histogramas + skewness), categóricas (barplots), temporal (hora/dia/mês)
4. Análise bivariada — categóricas × target (stacked barplot), numéricas × target (boxplot seaborn)
5. Matriz de correlação de Pearson + detecção automática de multicolinearidade (|r| > 0.8)
6. Informação Mútua (`mutual_info_classif`) — mais adequada que Pearson para target multi-classe
7. Conclusões parciais com valores reais

### Decisões técnicas tomadas
- `LIGHTCOND` convertida para string e tratada como categórica (é código ordinal, não numérico contínuo)
- Variáveis booleanas (`INATTENTIONIND`, `UNDERINFL`, `SPEEDING`, `HITPARKEDCAR`, `intersection_related`) incluídas na análise categórica
- `df.duplicated()` exclui `Unnamed: 0` e `SPDCASENO` (IDs sequenciais tornariam toda linha única por definição)
- Boxplots feitos com `sns.boxplot` — evita títulos automáticos indesejados do `df.boxplot`
- Pearson comentado como limitado para multi-classe; MI adicionada como complemento
- Categorias com < 50 ocorrências filtradas nos stacked barplots bivariados (proporções ruidosas)

### Resultados da EDA

**Target (SEVERITYCODE):**
| Classe | Contagem | Proporção |
|--------|----------|-----------|
| 0 | 71.979 | 64.33% |
| 1 | 37.804 | 33.79% |
| 2 | 1.909  | 1.71%  |
| 3 | 190    | 0.17%  |

- Dataset **fortemente desbalanceado** — classes 2 e 3 somam < 2%
- **Baseline de accuracy = 64.33%** (prever sempre classe 0)
- Métrica principal para avaliação: **F1-macro** (não accuracy)
- Estratégia: `class_weight='balanced'` nos modelos

**Valores ausentes:**
| Coluna | % Nulos | Decisão |
|--------|---------|---------|
| response_type / response_time | 86.55% | Remover |
| SNOW / SNWD | 24.19% | Imputar com 0 |
| WSF5 | 1.21% | Imputar com mediana |

**Duplicatas:** 19 linhas → remover no pré-processamento

**Leakage — remover da modelagem:**
- `INJURIES`, `SERIOUSINJURIES`, `FATALITIES` — são consequências do acidente, não preditores

**Multicolinearidade (|r| > 0.8):**
| Par | r | Decisão |
|-----|---|---------|
| TAVG × TMAX | 0.969 | Remover TMAX e TMIN, manter TAVG |
| TAVG × TMIN | 0.954 | |
| TMAX × TMIN | 0.875 | |
| AWND × WSF5 | 0.814 | Remover WSF5, manter AWND |

**Informação Mútua (top features):**
| Feature | MI Score |
|---------|----------|
| VEHCOUNT | 0.0823 |
| PEDCOUNT | 0.0427 |
| longitude | 0.0408 |
| latitude | 0.0404 |
| Variáveis climáticas | ≈ 0.000 |

**Encoding planejado para o Bloco 2:**
- Categóricas nominais (COLLISIONTYPE, WEATHER, ROADCOND, JUNCTIONTYPE) → One-Hot Encoding
- LIGHTCOND (ordinal) → manter como inteiro
- Booleanas → converter para int (0/1)

---

## Bloco 2 — `02_preprocessing.ipynb` (Pré-processamento)

### Etapas planejadas (baseadas nas conclusões do Bloco 1)
1. Remover colunas: `Unnamed: 0`, `SPDCASENO`, `DATE`, `TIME`, `response_type`, `response_time`, `TMAX`, `TMIN`, `WSF5`, `INJURIES`, `SERIOUSINJURIES`, `FATALITIES`
2. Remover 19 linhas duplicadas
3. Imputar nulos: SNOW/SNWD → 0; WSF5 → mediana (antes de remover WSF5); demais → mediana/moda conforme tipo
4. Tratar outliers (verificar PERSONCOUNT, PEDCOUNT, VEHCOUNT)
5. Converter booleanas para int (0/1)
6. One-Hot Encoding nas categóricas nominais
7. Normalização das features numéricas contínuas (StandardScaler — **apenas no treino, aplicar no teste**)
8. Split treino/teste estratificado (stratify=SEVERITYCODE) — 80/20

---

## Bloco 3 — `03_modeling.ipynb` (Modelagem e Avaliação)

### Modelos
1. **Regressão Logística** — baseline interpretável, `class_weight='balanced'`
2. **Random Forest** — ensemble, captura não-linearidade, `class_weight='balanced'`

### Métricas
- F1-macro (principal — adequado para desbalanceamento)
- Accuracy, Precision, Recall por classe
- Matriz de confusão
- Comparação direta entre modelos

### Discussão obrigatória
- Overfitting (treino vs teste)
- Limitações das features disponíveis
- Impacto do desbalanceamento

---

## Critérios de Avaliação (referência)

- **40% Conceitual:** qualidade das análises, domínio do conteúdo, interpretação dos resultados
- **30% Procedimental:** uso correto das técnicas, qualidade das visualizações
- **30% Atitudinal:** organização do GitHub/relatório, prazo, trabalho em equipe

## Restrições do Relatório

- Máximo 15 páginas
- Máximo 5 figuras
