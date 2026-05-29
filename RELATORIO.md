# Relatório Técnico: Predição da Severidade de Acidentes Rodoviários em Seattle

Trabalho 2 — T326 Ciência de Dados | LCDIA — Universidade de Fortaleza

---

## 1. Introdução e Compreensão dos Dados

### 1.1 Contextualização e Objetivo

A segurança rodoviária é um dos pilares da gestão urbana moderna. O objetivo deste projeto é desenvolver e avaliar modelos de Aprendizado de Máquina capazes de prever a severidade de acidentes rodoviários na cidade de Seattle, com base em fatores circunstanciais, temporais e ambientais. Uma ferramenta desse tipo permitiria às autoridades municipais e aos serviços de emergência antecipar cenários de alto risco e otimizar a alocação de recursos.

O conjunto de dados utilizado (*Vehicle Collision Data in Seattle, 2005–2019*) contém 111.882 registros e 34 colunas, abrangendo informações de localização, condições da via, clima, e características da colisão.

### 1.2 Análise do Target e Justificação do Baseline

A variável alvo `SEVERITYCODE` é um código **ordinal com 4 categorias** (0 a 3), em ordem crescente de severidade da colisão. Os modelos foram treinados sobre as 4 classes originais, sem remapeamento ou fusão. A análise de distribuição revelou um acentuado desbalanceamento de classes:

| Classe | Proporção |
|--------|-----------|
| 0 | 64,33% |
| 1 | 33,79% |
| 2 | 1,71% |
| 3 | 0,17% |

**Justificação do Baseline:** o modelo mais ingênuo possível (*Zero-R*), que previsse sempre a classe majoritária (Classe 0), obteria uma acurácia automática de **64,33%**. Portanto, qualquer modelo preditivo robusto deve superar esse limiar. Contudo, devido ao severo desbalanceamento e à importância crítica de detectar as classes 2 e 3, a métrica primária adotada é o **F1-Score Macro**, que atribui o mesmo peso ao desempenho de cada classe.

---

## 2. Preparação dos Dados e Engenharia de Atributos

### 2.1 Prevenção de Vazamento de Dados (Data Leakage)

Uma auditoria das variáveis identificou que `INJURIES` (número de feridos), `SERIOUSINJURIES` (feridos graves) e `FATALITIES` (número de mortos) constituem vazamento de dados. Esses atributos são preenchidos *a posteriori*, após a ocorrência e avaliação do acidente. Incluí-los na modelagem resultaria num modelo com desempenho artificialmente alto em treino, mas inútil na prática, pois essas informações não estão disponíveis no momento em que uma colisão é reportada. Por conseguinte, foram excluídos **antes da divisão dos dados**.

Adicionalmente, removeram-se variáveis com alta multicolinearidade (`TMAX`, `TMIN`, `WSF5` — correlação de Pearson |r| > 0.8 com `TAVG`/`AWND`), colunas com mais de 50% de valores ausentes (`response_type`, `response_time`) e identificadores sem valor preditivo (`Unnamed: 0`, `SPDCASENO`).

### 2.2 Tratamento de Dados e Estratégia de Codificação

- **Features temporais:** extraídas de `DATE` (mês, dia da semana) e `TIME` (hora do dia — armazenada em horas decimais).
- **Valores ausentes:** `SNOW` e `SNWD` imputados com 0 (ausência de neve é informativa); demais numéricas com a mediana e categóricas com a moda.
- **Outliers:** `PERSONCOUNT` e `VEHCOUNT` limitados ao percentil 99 (capping). `PEDCOUNT` e `PEDCYLCOUNT` foram mantidos integralmente, pois seus valores altos são raros mas legítimos e preditivos.
- **One-Hot Encoding** (`drop_first=True`): aplicado a `COLLISIONTYPE`, `WEATHER`, `ROADCOND` e `JUNCTIONTYPE`.
- **`LIGHTCOND`:** mantida como variável ordinal inteira (0–3), pois já codifica níveis crescentes de iluminação.
- **Variáveis booleanas** (`SPEEDING`, `UNDERINFL`, `INATTENTIONIND`, `HITPARKEDCAR`, `intersection_related`): convertidas para inteiro (0/1).
- **Padronização:** `StandardScaler` aplicado às features contínuas e ordinais, com `fit` realizado **apenas no conjunto de treino** e `transform` aplicado a treino e teste, evitando vazamento estatístico.

A divisão do conjunto seguiu a proporção de **80% para treino e 20% para teste**, de forma **estratificada** (`stratify=y`), preservando a distribuição de classes nos dois conjuntos.

---

## 3. Modelagem e Treino

Para o problema de classificação multiclasse desbalanceada, foram selecionados dois algoritmos de naturezas distintas:

1. **Regressão Logística (Multinomial):** modelo linear baseline que estima as probabilidades das classes via função *softmax*. Configurado com `max_iter=1000` para garantir convergência.
2. **Random Forest Classifier:** modelo não-linear baseado num *ensemble* de 100 árvores de decisão construídas por *bagging*, robusto a interações complexas entre atributos.

**Tratamento do Desbalanceamento:** ambos os algoritmos foram parametrizados com `class_weight='balanced'`, que penaliza erros nas classes minoritárias de forma inversamente proporcional à sua frequência, forçando a fronteira de decisão a considerar ferimentos e fatalidades.

---

## 4. Análise de Resultados e Comparação

### Tabela 1: Desempenho Comparativo (Treino vs. Teste)

| Métrica | LR (Treino) | LR (Teste) | RF (Treino) | RF (Teste) |
|---------|-------------|------------|-------------|------------|
| Acurácia | 0,5647 | 0,5597 | 1,0000 | 0,7152 |
| Precisão (Macro) | 0,3393 | 0,3357 | 1,0000 | 0,3451 |
| Recall (Macro) | 0,5332 | 0,4778 | 1,0000 | 0,3238 |
| F1-Score (Macro) | 0,3269 | 0,3219 | 1,0000 | 0,3239 |

**Figura 1:** Matrizes de confusão dos dois modelos no conjunto de teste (geradas no Bloco 3 — recomenda-se juntar os dois gráficos lado a lado para contar como uma única figura).

### 4.1 Diagnóstico de Overfitting e Análise Crítica

- **Regressão Logística:** apresentou consistência notável entre treino e teste (F1-Macro de 0,3269 vs. 0,3219). O gap praticamente nulo demonstra **ausência de overfitting**. Contudo, o valor absoluto baixo indica **underfitting estrutural**: por ser linear, o modelo não captura combinações complexas entre atributos. A acurácia (0,56) fica *abaixo* do baseline de 64,33%, pois o `class_weight='balanced'` troca acurácia global por maior sensibilidade às classes minoritárias.
- **Random Forest:** apresentou **overfitting severo**. No treino, todas as métricas atingiram 1,0000 (memorização perfeita do conjunto), mas o F1-Macro colapsou para 0,3239 no teste. As árvores cresceram sem restrição de profundidade (`max_depth`), decorando especificidades e ruídos do treino que não generalizam. A acurácia de teste (0,7152) supera o baseline, mas o F1-Macro revela que o ganho concentra-se nas classes majoritárias.

### 4.2 Desempenho nas Classes Minoritárias

O `class_weight='balanced'` aumentou o recall das classes 2 e 3 em relação a um modelo padrão. A Regressão Logística obteve Recall Macro de teste **0,4778**, superior ao Random Forest (**0,3238**), mostrando-se mais eficaz em identificar acidentes com feridos e fatalidades — ainda que gerando mais falsos positivos devido à sua simplicidade linear. As matrizes de confusão evidenciam que ambos os modelos ainda confundem bastante as classes raras, que somam menos de 2% dos dados.

---

## 5. Limitações e Trabalho Futuro

### 5.1 Limitações Identificadas

1. **Remoção de variáveis críticas:** a exclusão obrigatória das variáveis de vazamento (`INJURIES`, `SERIOUSINJURIES`, `FATALITIES`) removeu os indicadores diretos de severidade. Prever a gravidade usando apenas variáveis de contexto (clima, via, tipo de colisão) mostrou-se um desafio, dado que o comportamento humano (velocidade, uso de cinto, sobriedade) — frequentemente não registrado de forma estruturada no momento inicial — determina a gravidade real do impacto. Isso é confirmado pela baixa Informação Mútua das features disponíveis com o target (todas < 0.09).
2. **Subotimização de hiperparâmetros:** o Random Forest não sofreu poda (*pruning*), resultando no overfitting documentado.
3. **Desbalanceamento extremo:** as classes 2 e 3 (< 2% dos dados) limitam o aprendizado dos padrões de acidentes graves.

### 5.2 Sugestões de Trabalho Futuro

- **Regularização do Random Forest:** limitar `max_depth` (ex.: 8–15) e aumentar `min_samples_leaf` para controlar o overfitting.
- **Modelos de Gradient Boosting:** implementar XGBoost ou LightGBM, que corrigem erros sequencialmente e lidam melhor com desbalanceamento.
- **Reamostragem:** aplicar SMOTE ou undersampling para reforçar as classes minoritárias.
- **Engenharia de atributos avançada:** criar variáveis de interação (ex.: `IS_NIGHT_AND_RAIN`) e componentes temporais mais ricas (horas de pico, fins de semana).
- **Tunagem de hiperparâmetros** via validação cruzada.

---

## Referências

- Dataset: *Vehicle Collision Data in Seattle (2005–2019)* — Kaggle.
- Documentação scikit-learn: https://scikit-learn.org
