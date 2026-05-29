import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

sns.set_theme(style="whitegrid")
os.makedirs("figuras", exist_ok=True)

df = pd.read_csv("data/seattle_collision_data_2005_2019.csv")

rotulos = {0: "0\nDanos\nmateriais", 1: "1\nFerimentos", 2: "2\nFerimentos\ngraves", 3: "3\nFatalidade"}
contagem = df["SEVERITYCODE"].value_counts().sort_index()
prop = df["SEVERITYCODE"].value_counts(normalize=True).sort_index() * 100

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar([rotulos[i] for i in contagem.index], contagem.values,
              color=sns.color_palette("Blues_d", len(contagem)))
ax.set_ylabel("Quantidade de registros")
ax.set_title("Figura 1 — Distribuição da variável alvo (SEVERITYCODE)")
for bar, p in zip(bars, prop.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 800,
            f"{p:.2f}%", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("figuras/fig1_distribuicao_alvo.png", dpi=130)
plt.close()

LEAKAGE = ["INJURIES", "SERIOUSINJURIES", "FATALITIES"]
EXCLUIR = ["Unnamed: 0", "SEVERITYCODE", "LIGHTCOND"] + LEAKAGE
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in EXCLUIR]

corr = df[num_cols + ["SEVERITYCODE"]].corr()
fig, ax = plt.subplots(figsize=(11, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, annot_kws={"size": 7}, ax=ax)
ax.set_title("Figura 2 — Matriz de Correlação de Pearson (variáveis numéricas)")
plt.tight_layout()
plt.savefig("figuras/fig2_correlacao.png", dpi=130)
plt.close()

X_mi = df[num_cols].fillna(df[num_cols].median(numeric_only=True))
mi = pd.Series(mutual_info_classif(X_mi, df["SEVERITYCODE"], random_state=42),
               index=num_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 4.5))
mi.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white")
ax.set_ylabel("Informação Mútua")
ax.set_title("Figura 3 — Informação Mútua das features com o alvo")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("figuras/fig3_informacao_mutua.png", dpi=130)
plt.close()

data = df.copy()
data["DATE"] = pd.to_datetime(data["DATE"], errors="coerce")
data["mes"] = data["DATE"].dt.month
data["dia_da_semana"] = data["DATE"].dt.dayofweek
data["hora"] = data["TIME"].fillna(data["TIME"].median()).astype(int)
data = data.drop(columns=["Unnamed: 0", "SPDCASENO", "DATE", "TIME"] + LEAKAGE +
                 ["TMAX", "TMIN", "WSF5", "response_type", "response_time"], errors="ignore")
data = data.drop_duplicates().reset_index(drop=True)
data["SNOW"] = data["SNOW"].fillna(0)
data["SNWD"] = data["SNWD"].fillna(0)
for c in data.select_dtypes(include=[np.number]).columns:
    if data[c].isnull().any():
        data[c] = data[c].fillna(data[c].median())
for c in data.select_dtypes(include=["str", "object", "category"]).columns:
    if data[c].isnull().any():
        data[c] = data[c].fillna(data[c].mode()[0])
for c in ["PERSONCOUNT", "VEHCOUNT"]:
    data[c] = data[c].clip(upper=data[c].quantile(0.99))
bc = data.select_dtypes(include=["bool"]).columns
data[bc] = data[bc].astype(int)
data = pd.get_dummies(data, columns=["COLLISIONTYPE", "WEATHER", "ROADCOND", "JUNCTIONTYPE"], drop_first=True)
oc = data.select_dtypes(include=["bool"]).columns
data[oc] = data[oc].astype(int)
data["LIGHTCOND"] = data["LIGHTCOND"].astype(int)
X = data.drop("SEVERITYCODE", axis=1)
y = data["SEVERITYCODE"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cont = [c for c in X_train.columns if X_train[c].nunique() > 2]
scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[cont] = scaler.fit_transform(X_train[cont])
X_test[cont] = scaler.transform(X_test[cont])

lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, modelo, nome in [(axes[0], lr, "Regressão Logística"), (axes[1], rf, "Random Forest")]:
    cm = confusion_matrix(y_test, modelo.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()))
    ax.set_title(nome)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
fig.suptitle("Figura 4 — Matrizes de Confusão no conjunto de teste", fontsize=13)
plt.tight_layout()
plt.savefig("figuras/fig4_matrizes_confusao.png", dpi=130)
plt.close()

print("Figuras geradas em figuras/:", os.listdir("figuras"))
