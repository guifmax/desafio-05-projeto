import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Carregar o arquivo Excel
# Caso queira utilizar este código, altere o diretório com o caminho do arquivo da tabela.
caminho_excel = r"D:\Users\gfm3\Downloads\Failing_Equipment_Exercise.xlsx"
df = pd.read_excel(caminho_excel, header=0, index_col=0)

# 2. Calcular estatísticas do código original
desvio_padrao_por_objeto = df.std(axis=0)
media_por_objeto = df.mean(axis=0)
desvio_padrao_por_banda = df.std(axis=1)
media_por_banda = df.mean(axis=1)
z_scores_por_banda = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
outliers = z_scores_por_banda.abs() > 2

# 3. Imprimir estatísticas (do código original)
print("Desvio Padrão por Objeto:")
print(desvio_padrao_por_objeto)
print("\nMédias por Objeto:")
print(media_por_objeto)
print("\nDesvio Padrão por Banda:")
print(desvio_padrao_por_banda)
print("\nMédias por Banda:")
print(media_por_banda)
print("\nZ-Scores por Banda:\n", z_scores_por_banda)
print("\nOutliers por Banda:\n", outliers[outliers].stack())

# 4. Dados de Z-Scores fornecidos (para novos gráficos)
z_scores = pd.DataFrame({
    "Objeto 1": [-0.21, -0.91, -0.50, 2.21, -0.81, -0.92, -0.61, -0.71, 1.10, 0.55, -0.48, 0.48, 0.04, -0.60, -0.24, 2.70, 0.69],
    "Objeto 2": [-0.61, -0.80, -0.34, 2.22, -0.76, -0.81, -0.46, -0.70, 0.51, 1.76, -0.60, 0.33, -0.16, -0.51, -0.20, 2.73, 0.61],
    "Objeto 3": [-0.00, -0.87, -0.46, 2.13, -0.76, -0.85, -0.59, -0.72, 1.06, 0.23, -0.62, 0.62, -0.09, -0.51, -0.26, 2.77, 0.67],
    "Objeto 4": [-0.06, -0.93, -0.60, 2.32, -0.86, -0.95, -0.58, -0.74, 1.56, 0.80, -0.52, 0.64, 0.14, -0.65, -0.30, 1.62, 0.71]
}, index=[f"Banda {i}" for i in range(1, 18)])

# 5. Dados brutos para dendrograma (do código fornecido)
df_dendrograma = pd.DataFrame({
    "Banda 1": [375, 135, 458, 475],
    "Banda 2": [57, 47, 53, 73],
    "Banda 3": [245, 267, 242, 227],
    "Banda 4": [1472, 1494, 1462, 1582],
    "Banda 5": [105, 66, 103, 103],
    "Banda 6": [54, 41, 62, 64],
    "Banda 7": [193, 209, 184, 235],
    "Banda 8": [147, 93, 122, 160],
    "Banda 9": [1102, 674, 957, 1137],
    "Banda 10": [720, 1033, 566, 874],
    "Banda 11": [253, 143, 171, 265],
    "Banda 12": [685, 586, 750, 803],
    "Banda 13": [488, 355, 418, 570],
    "Banda 14": [198, 187, 220, 203],
    "Banda 15": [360, 334, 337, 365],
    "Banda 16": [1374, 1506, 1572, 1256],
    "Banda 17": [156, 139, 147, 175]
}, index=["Objeto 1", "Objeto 2", "Objeto 3", "Objeto 4"])

# 6. Gráfico de Desvio Padrão e Média por Objeto
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
desvio_padrao_por_objeto.plot(kind='bar', color='skyblue')
plt.title('Desvio Padrão por Objeto')
plt.xlabel('Objeto')
plt.ylabel('Desvio Padrão')
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
media_por_objeto.plot(kind='bar', color='lightgreen')
plt.title('Média por Objeto')
plt.xlabel('Objeto')
plt.ylabel('Média')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. Gráfico de Desvio Padrão e Média por Banda
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
desvio_padrao_por_banda.plot(kind='bar', color='salmon')
plt.title('Desvio Padrão por Banda')
plt.xlabel('Banda')
plt.ylabel('Desvio Padrão')
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
media_por_banda.plot(kind='bar', color='lightblue')
plt.title('Média por Banda')
plt.xlabel('Banda')
plt.ylabel('Média')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8. Mapa de Calor de Z-Scores por Banda
plt.figure(figsize=(10, 8))
sns.heatmap(z_scores, cmap='coolwarm', center=0, annot=True, fmt='.2f')
plt.title('Mapa de Calor de Z-Scores por Banda')
plt.xlabel('Objetos')
plt.ylabel('Bandas')
plt.tight_layout()
plt.show()

# 9. Dendrograma
Z = linkage(df_dendrograma, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=df_dendrograma.index, leaf_rotation=90)
plt.title("Dendrograma de Objetos")
plt.xlabel("Objetos")
plt.ylabel("Distância")
plt.tight_layout()
plt.show()

# 10. Distribuição de Z-Scores por Objeto
plt.figure(figsize=(12, 6))
for coluna in z_scores.columns:
    sns.kdeplot(z_scores[coluna], label=coluna, linewidth=2 if coluna == "Objeto 2" else 1,
                alpha=1 if coluna == "Objeto 2" else 0.5)
plt.title("Distribuição de Z-Scores por Objeto")
plt.xlabel("Z-Score")
plt.ylabel("Densidade")
plt.legend()
plt.tight_layout()
plt.show()

# 11. Z-Scores por Banda com Objeto 2 destacado
plt.figure(figsize=(12, 6))
for coluna in z_scores.columns:
    plt.plot(z_scores.index, z_scores[coluna], marker='o', label=coluna,
             linewidth=2 if coluna == "Objeto 2" else 1, alpha=1 if coluna == "Objeto 2" else 0.5)
plt.title("Z-Scores por Banda (Destaque: Objeto 2)")
plt.xlabel("Bandas")
plt.ylabel("Z-Score")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


