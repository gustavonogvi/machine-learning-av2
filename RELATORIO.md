# Relatório Técnico: Predição da Severidade de Acidentes Rodoviários em Seattle

Trabalho 2 — T326 Ciência de Dados | LCDIA — Universidade de Fortaleza

---

## 1. Introdução

### 1.1 Descrição do Problema

A segurança rodoviária é um dos pilares da gestão urbana moderna. Acidentes de trânsito geram custos humanos e econômicos elevados, e a capacidade de antecipar a gravidade de uma colisão a partir das suas circunstâncias permite priorizar respostas de emergência e ações de prevenção. O problema central deste trabalho é: **dado o conjunto de condições em que uma colisão ocorre (local, clima, via, iluminação, tipo de colisão, envolvidos), é possível prever a sua severidade?**

### 1.2 Objetivo do Trabalho

Desenvolver e avaliar modelos de Aprendizado de Máquina capazes de classificar a severidade de acidentes rodoviários na cidade de Seattle, com base em fatores circunstanciais, temporais e ambientais, aplicando um fluxo completo de ciência de dados: análise exploratória, pré-processamento rigoroso (com prevenção de vazamento de dados), modelagem com pelo menos dois algoritmos e avaliação crítica com métricas adequadas ao desbalanceamento.

---

## 2. Dataset

### 2.1 Fonte dos Dados

Conjunto de dados *Vehicle Collision Data in Seattle (2005–2019)*, disponível na plataforma Kaggle, contendo o histórico de colisões registradas pela cidade de Seattle. O conjunto possui **111.882 registros** e **34 colunas**.

### 2.2 Tipo do Problema

Trata-se de um problema de **classificação supervisionada multiclasse**, em que o alvo `SEVERITYCODE` é um código ordinal com 4 níveis crescentes de severidade (0 a 3).

### 2.3 Descrição das Principais Variáveis

| Grupo | Variáveis | Descrição |
|-------|-----------|-----------|
| **Alvo** | `SEVERITYCODE` | Severidade da colisão (0 a 3, ordinal crescente) |
| **Localização** | `longitude`, `latitude` | Coordenadas geográficas do acidente |
| **Características** | `COLLISIONTYPE`, `JUNCTIONTYPE` | Tipo de colisão e tipo de cruzamento/via |
| **Envolvidos** | `PERSONCOUNT`, `PEDCOUNT`, `PEDCYLCOUNT`, `VEHCOUNT` | Contagem de pessoas, pedestres, ciclistas e veículos |
| **Comportamento** | `SPEEDING`, `UNDERINFL`, `INATTENTIONIND`, `HITPARKEDCAR`, `intersection_related` | Indicadores booleanos (excesso de velocidade, sob influência, desatenção, etc.) |
| **Condições** | `WEATHER`, `ROADCOND`, `LIGHTCOND` | Clima, condição da via e iluminação no momento |
| **Meteorologia** | `AWND`, `PRCP`, `SNOW`, `SNWD`, `TAVG`, `TMAX`, `TMIN`, `WSF5` | Vento, precipitação, neve e temperatura |
| **Temporais** | `DATE`, `TIME` | Data e hora do acidente |
| **Pós-acidente** | `INJURIES`, `SERIOUSINJURIES`, `FATALITIES` | Consequências do acidente (não usadas — ver 3.2) |

---

## 3. Análise e Pré-processamento

### 3.1 Análise Exploratória dos Dados (EDA)

**Distribuição do alvo:** a análise revelou forte desbalanceamento de classes:

| Classe | Proporção |
|--------|-----------|
| 0 | 64,33% |
| 1 | 33,79% |
| 2 | 1,71% |
| 3 | 0,17% |

As classes 2 e 3 somam menos de 2% dos dados. Isso define o **baseline**: prever sempre a classe majoritária (Classe 0) daria acurácia de **64,33%** — limiar que qualquer modelo deve superar. Pela criticidade das classes graves, a métrica primária adotada é o **F1-Score Macro**, que pesa igualmente todas as classes. *(Figura 1: distribuição do alvo.)*

**Correlações e multicolinearidade:** a matriz de correlação de Pearson identificou pares de variáveis fortemente correlacionadas entre si (|r| > 0,8): `TAVG`×`TMAX` (0,969), `TAVG`×`TMIN` (0,954), `TMAX`×`TMIN` (0,875) e `AWND`×`WSF5` (0,814) — indicando redundância a ser tratada. *(Figura 2: heatmap de correlação.)*

**Relevância das features:** como a correlação de Pearson é limitada para alvos multiclasse, calculou-se a **Informação Mútua** entre cada variável e o alvo. As mais informativas foram `VEHCOUNT` (0,082), `PEDCOUNT` (0,043), `longitude` (0,041) e `latitude` (0,040). As variáveis meteorológicas apresentaram informação mútua próxima de zero, indicando baixo poder preditivo isolado. *(Figura 3: ranking de Informação Mútua.)*

**Valores ausentes:** identificaram-se nulos relevantes em `response_type`/`response_time` (86,55%), `SNOW`/`SNWD` (24,19%) e `WSF5` (1,21%).

### 3.2 Prevenção de Vazamento de Dados (Data Leakage)

Identificou-se que `INJURIES`, `SERIOUSINJURIES` e `FATALITIES` constituem **vazamento de dados**: são preenchidas *a posteriori*, após a avaliação do acidente. Incluí-las geraria um modelo com desempenho artificial em treino e inútil na prática, pois não estão disponíveis no momento do registro inicial. Foram excluídas **antes da divisão dos dados**.

### 3.3 Tratamentos Realizados

- **Colunas removidas:** variáveis de vazamento; multicolineares (`TMAX`, `TMIN`, `WSF5`); com >50% de nulos (`response_type`, `response_time`); e identificadores (`Unnamed: 0`, `SPDCASENO`).
- **Duplicatas:** registros integralmente duplicados foram removidos.
- **Valores ausentes:** `SNOW` e `SNWD` imputados com 0 (ausência de neve é informativa); demais numéricas com a mediana e categóricas com a moda.
- **Outliers:** `PERSONCOUNT` e `VEHCOUNT` limitados ao percentil 99 (capping). `PEDCOUNT` e `PEDCYLCOUNT` mantidos, pois seus valores altos são raros mas legítimos e preditivos.

### 3.4 Transformações Aplicadas

- **Features temporais:** extraídas de `DATE` (mês, dia da semana) e `TIME` (hora do dia, armazenada em horas decimais).
- **One-Hot Encoding** (`drop_first=True`): `COLLISIONTYPE`, `WEATHER`, `ROADCOND`, `JUNCTIONTYPE`.
- **`LIGHTCOND`:** mantida como variável ordinal inteira (0–3), pois já codifica níveis crescentes de iluminação.
- **Booleanas:** convertidas para inteiro (0/1).
- **Padronização:** `StandardScaler` nas features contínuas e ordinais, com `fit` **apenas no treino** e `transform` em treino e teste, evitando vazamento estatístico.
- **Divisão:** 80% treino / 20% teste, **estratificada** (`stratify=y`), preservando a distribuição de classes.

---

## 4. Modelagem

### 4.1 Modelos Utilizados

Foram selecionados dois algoritmos de naturezas distintas:

1. **Regressão Logística (Multinomial):** modelo linear baseline que estima as probabilidades das classes via função *softmax*. Configurado com `max_iter=1000` para garantir convergência. Vantagem: interpretabilidade.
2. **Random Forest Classifier:** modelo não-linear baseado em *ensemble* de 100 árvores de decisão construídas por *bagging*, capaz de capturar interações complexas entre atributos.

### 4.2 Estratégia de Treinamento

Ambos os algoritmos foram parametrizados com `class_weight='balanced'`, que penaliza erros nas classes minoritárias de forma inversamente proporcional à sua frequência, forçando a fronteira de decisão a considerar ferimentos e fatalidades. O treino foi feito sobre o conjunto padronizado de treino (80%) e a avaliação sobre o conjunto de teste (20%) independente. Usou-se `random_state=42` para reprodutibilidade.

---

## 5. Resultados

### 5.1 Métricas Utilizadas

A métrica primária é o **F1-Score Macro** (adequado ao desbalanceamento). Reportam-se também Acurácia, Precisão e Recall (macro), além das matrizes de confusão.

### 5.2 Comparação entre Modelos

**Tabela 1: Desempenho Comparativo (Treino vs. Teste)**

| Métrica | LR (Treino) | LR (Teste) | RF (Treino) | RF (Teste) |
|---------|-------------|------------|-------------|------------|
| Acurácia | 0,5647 | 0,5597 | 1,0000 | 0,7152 |
| Precisão (Macro) | 0,3393 | 0,3357 | 1,0000 | 0,3451 |
| Recall (Macro) | 0,5332 | 0,4778 | 1,0000 | 0,3238 |
| F1-Score (Macro) | 0,3269 | 0,3219 | 1,0000 | 0,3239 |

*(Figura 4: matrizes de confusão dos dois modelos no conjunto de teste — recomenda-se juntar os dois gráficos lado a lado.)*

### 5.3 Discussão dos Resultados

- **Regressão Logística:** consistência notável entre treino e teste (F1-Macro 0,3269 vs. 0,3219) — gap praticamente nulo, **ausência de overfitting**. Porém, o valor absoluto baixo indica **underfitting estrutural**: por ser linear, não captura combinações complexas. A acurácia (0,56) fica *abaixo* do baseline porque o `class_weight='balanced'` troca acurácia global por sensibilidade às classes minoritárias.
- **Random Forest:** **overfitting severo** — todas as métricas atingem 1,0000 no treino (memorização perfeita), mas o F1-Macro colapsa para 0,3239 no teste. As árvores cresceram sem restrição de profundidade. A acurácia de teste (0,7152) supera o baseline, mas o F1-Macro revela que o ganho concentra-se nas classes majoritárias.
- **Classes minoritárias:** a Regressão Logística obteve Recall Macro de teste (0,4778) superior ao Random Forest (0,3238), sendo mais eficaz em identificar acidentes com feridos e fatalidades, ainda que gerando mais falsos positivos. As matrizes de confusão mostram que ambos ainda confundem bastante as classes raras.

---

## 6. Conclusão

### 6.1 Principais Aprendizados

O trabalho demonstrou um fluxo completo de ciência de dados aplicado a um problema real e desafiador. Os principais aprendizados foram:

- A **prevenção de vazamento de dados** é decisiva: incluir as variáveis de consequência (`INJURIES`, `FATALITIES`) inflaria artificialmente o desempenho, mas tornaria o modelo inútil na prática.
- Em conjuntos **fortemente desbalanceados**, a acurácia é enganosa — o F1-Score Macro e a análise por classe revelam o desempenho real, especialmente nas classes críticas.
- A comparação entre um modelo linear (Regressão Logística) e um *ensemble* (Random Forest) evidenciou de forma didática os fenômenos opostos de **underfitting** e **overfitting**.
- As features puramente circunstanciais disponíveis têm **baixo poder preditivo** sobre a severidade (confirmado pela Informação Mútua), o que limita o teto de desempenho alcançável.

### 6.2 Limitações e Possíveis Melhorias

**Limitações:**

1. A remoção obrigatória das variáveis de vazamento retirou os indicadores diretos de severidade; o comportamento humano (velocidade real, uso de cinto, sobriedade) — não registrado estruturadamente no momento inicial — é o que de fato determina a gravidade.
2. O Random Forest não sofreu poda, resultando no overfitting documentado.
3. O desbalanceamento extremo (classes 2 e 3 < 2%) limita o aprendizado dos padrões graves.

**Possíveis melhorias:**

- Regularizar o Random Forest (`max_depth`, `min_samples_leaf`).
- Explorar modelos de *Gradient Boosting* (XGBoost, LightGBM), mais robustos a desbalanceamento.
- Aplicar técnicas de reamostragem (SMOTE, undersampling).
- Engenharia de atributos avançada (variáveis de interação como "noite + chuva", horas de pico, fins de semana).
- Tunagem de hiperparâmetros via validação cruzada.

---

## 7. Referências

- Dataset: *Vehicle Collision Data in Seattle (2005–2019)* — Kaggle. Disponível em: https://www.kaggle.com/datasets/mcfly1/vehicle-collision-data-in-seattle-2005-2019
- Pedregosa et al. *Scikit-learn: Machine Learning in Python*. JMLR, 2011. https://scikit-learn.org
- McKinney, W. *Data Structures for Statistical Computing in Python* (pandas), 2010.
