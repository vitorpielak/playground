# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

#EXPLORAÇÃO INICIAL E LIMPEZA DOS DADOS

#carregamento de dados
df = pd.read_csv("Hearing.csv",delimiter=",",encoding="latin-1")

#inicio
print("Shape dos dados:",df.shape[0],"Linhas e",df.shape[1],"Colunas")
#%%
print("\nColunas do dataset:")
print(df.columns.to_list())
#%%
print("Primeiras 5 linhas")
print(df.head())
#%%
#tipos dos dados
print("\nTipos dos dados:")
print(df.dtypes)

# %%
#verificar nulos
print('\n Valores nulos por coluna:')
print(df.isnull().sum())
# %%
#estatisticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe())

# %%
#Limpeza
# LIMPEZA E PADRONIZAÇÃO DOS DADO

# Criar cópia dos dados originais
df_clean = df.copy()

# Renomear colunas para facilitar análise (nomes mais curtos e claros)
column_mapping = {
    'Perceived_Hearing_Meaning': 'hearing_meaning',
    'Hearing_FOMO': 'hearing_fomo',
    'Hearing_Test_Barrier': 'test_barrier',
    'Missed_Important_Sounds': 'missed_sounds',
    'Left_Out_Due_To_Hearing': 'left_out',
    'Daily_Headphone_Use': 'headphone_use',
    'Belief_Early_Hearing_Care': 'early_care_belief',
    'Last_Hearing_Test_Method': 'last_test_method',
    'Interest_in_Hearing_App': 'app_interest',
    'Desired_App_Features': 'desired_features',
    'Awareness_on_hearing_and_Willingness_to_invest': 'willingness_invest',
    'Paid_App_Test_Interest': 'paid_app_interest',
    'Age_group': 'age_group',
    'Ear_Discomfort_After_Use': 'ear_discomfort'
}

df_clean = df_clean.rename(columns=column_mapping)

print("✅ Colunas renomeadas com sucesso!")

# Análise de distribuições principais
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Distribuições das Principais Variáveis Categóricas', fontsize=16, fontweight='bold')

# 1. Age groups
df_clean['age_group'].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Distribuição por Faixa Etária')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. App interest
df_clean['app_interest'].value_counts().plot(kind='bar', ax=axes[0,1], color='lightgreen')
axes[0,1].set_title('Interesse em App de Audição')
axes[0,1].tick_params(axis='x', rotation=45)

# 3. Headphone use
df_clean['headphone_use'].value_counts().plot(kind='bar', ax=axes[0,2], color='coral')
axes[0,2].set_title('Uso Diário de Fones')
axes[0,2].tick_params(axis='x', rotation=45)

# 4. Hearing FOMO
df_clean['hearing_fomo'].value_counts().plot(kind='bar', ax=axes[1,0], color='gold')
axes[1,0].set_title('FOMO Auditivo')
axes[1,0].tick_params(axis='x', rotation=45)

# 5. Last test method
df_clean['last_test_method'].value_counts().plot(kind='bar', ax=axes[1,1], color='plum')
axes[1,1].set_title('Último Método de Teste')
axes[1,1].tick_params(axis='x', rotation=45)

# 6. Ear discomfort
df_clean['ear_discomfort'].value_counts().plot(kind='bar', ax=axes[1,2], color='lightcoral')
axes[1,2].set_title('Desconforto no Ouvido')
axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("\n📊 Gráficos de distribuição gerados!")
print(f"\n✅ LIMPEZA CONCLUÍDA!")
print(f"Dataset limpo salvo como 'df_clean' com {df_clean.shape[0]} registros e {df_clean.shape[1]} variáveis")


# %%

#ANÁLISE EXPLORATÓRIA APROFUNDADA



print("="*80)
print("ANÁLISE EXPLORATÓRIA APROFUNDADA")
print("="*80)

# 1. ANÁLISE DE CORRELAÇÕES E RELACIONAMENTOS
print("\n🔍 1. ANÁLISE DE RELACIONAMENTOS ENTRE VARIÁVEIS")

# Criar variáveis numéricas para análise de correlação
df_numeric = df_clean.copy()

# Encoding das principais variáveis categóricas para análise numérica
categorical_cols = ['hearing_fomo', 'app_interest', 'headphone_use', 'ear_discomfort', 
                   'age_group', 'last_test_method', 'willingness_invest', 'paid_app_interest']

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_numeric[f'{col}_encoded'] = le.fit_transform(df_numeric[col].astype(str))
    le_dict[col] = le
    print(f"✅ {col} codificado: {len(le.classes_)} categorias")

# Matriz de correlação
numeric_cols = [col for col in df_numeric.columns if col.endswith('_encoded')] + ['early_care_belief']
correlation_matrix = df_numeric[numeric_cols].corr()

# Visualizar matriz de correlação
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
plt.title('Matriz de Correlação - Variáveis Principais', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n📈 Correlações mais relevantes encontradas:")
# Extrair correlações mais altas (excluindo diagonal)
corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.3:  # Correlações moderadas ou fortes
            corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))

corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for col1, col2, corr in corr_pairs[:5]:
    print(f"• {col1} ↔ {col2}: {corr:.3f}")

# 2. ANÁLISE DEMOGRÁFICA APROFUNDADA
print(f"\n👥 2. SEGMENTAÇÃO DEMOGRÁFICA")

# Interesse em app por faixa etária
interest_age = pd.crosstab(df_clean['age_group'], df_clean['app_interest'], normalize='index') * 100

plt.figure(figsize=(12, 6))
interest_age.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Interesse em App de Audição por Faixa Etária (%)', fontsize=14, fontweight='bold')
plt.xlabel('Faixa Etária')
plt.ylabel('Percentual (%)')
plt.legend(title='Interesse em App', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("📊 Análise por faixa etária:")
for age in df_clean['age_group'].unique():
    if pd.notna(age):
        subset = df_clean[df_clean['age_group'] == age]
        interested = len(subset[subset['app_interest'] == 'Yes, that would be helpful'])
        total = len(subset)
        print(f"• {age}: {interested}/{total} ({interested/total*100:.1f}%) interessados em app")

# 3. PADRÕES DE USO DE FONES E DESCONFORTO
print(f"\n🎧 3. ANÁLISE DE USO DE FONES vs DESCONFORTO")

# Crosstab uso de fones vs desconforto
headphone_discomfort = pd.crosstab(df_clean['headphone_use'], df_clean['ear_discomfort'])

plt.figure(figsize=(10, 6))
headphone_discomfort_pct = pd.crosstab(df_clean['headphone_use'], df_clean['ear_discomfort'], normalize='index') * 100
headphone_discomfort_pct.plot(kind='bar', ax=plt.gca())
plt.title('Desconforto no Ouvido por Tempo de Uso de Fones (%)', fontsize=14, fontweight='bold')
plt.xlabel('Tempo de Uso Diário de Fones')
plt.ylabel('Percentual (%)')
plt.legend(title='Desconforto', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("📊 Relação uso de fones vs desconforto:")
for headphone in df_clean['headphone_use'].unique():
    if pd.notna(headphone):
        subset = df_clean[df_clean['headphone_use'] == headphone]
        discomfort_yes = len(subset[subset['ear_discomfort'].isin(['Yes', 'Occasionally'])])
        total = len(subset)
        print(f"• {headphone}: {discomfort_yes}/{total} ({discomfort_yes/total*100:.1f}%) com desconforto")

# 4. ANÁLISE DE BARREIRAS PARA TESTES AUDITIVOS
print(f"\n🚧 4. ANÁLISE DE BARREIRAS PARA TESTES AUDITIVOS")

# Contar barreiras mais comuns
barriers = df_clean['test_barrier'].value_counts()
print("Top 5 barreiras para testes auditivos:")
for i, (barrier, count) in enumerate(barriers.head().items(), 1):
    print(f"{i}. {barrier}: {count} ({count/len(df_clean)*100:.1f}%)")

# Visualizar barreiras
plt.figure(figsize=(12, 6))
barriers.head(8).plot(kind='bar', color='lightcoral')
plt.title('Principais Barreiras para Testes Auditivos', fontsize=14, fontweight='bold')
plt.xlabel('Barreiras')
plt.ylabel('Número de Respondentes')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 5. ANÁLISE DE FOMO AUDITIVO
print(f"\n😰 5. ANÁLISE DE FOMO AUDITIVO")

# FOMO vs interesse em app
fomo_app = pd.crosstab(df_clean['hearing_fomo'], df_clean['app_interest'], normalize='index') * 100

plt.figure(figsize=(10, 6))
fomo_app.plot(kind='bar', ax=plt.gca())
plt.title('FOMO Auditivo vs Interesse em App (%)', fontsize=14, fontweight='bold')
plt.xlabel('Frequência de FOMO Auditivo')
plt.ylabel('Percentual (%)')
plt.legend(title='Interesse em App', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("📊 FOMO auditivo por nível:")
for fomo in df_clean['hearing_fomo'].unique():
    if pd.notna(fomo):
        subset = df_clean[df_clean['hearing_fomo'] == fomo]
        interested = len(subset[subset['app_interest'] == 'Yes, that would be helpful'])
        total = len(subset)
        print(f"• {fomo}: {interested}/{total} ({interested/total*100:.1f}%) interessados em app")

# 6. ANÁLISE TEXTUAL - SIGNIFICADO DA AUDIÇÃO
print(f"\n📝 6. ANÁLISE TEXTUAL: SIGNIFICADO DA AUDIÇÃO")

# Extrair e contar temas mais comuns no significado da audição
meaning_themes = []
for meaning in df_clean['hearing_meaning']:
    if pd.notna(meaning):
        themes = [theme.strip() for theme in meaning.split(',')]
        meaning_themes.extend(themes)

meaning_counts = pd.Series(meaning_themes).value_counts()
print("Top 10 significados da audição:")
for i, (theme, count) in enumerate(meaning_counts.head(10).items(), 1):
    print(f"{i:2d}. {theme}: {count} ({count/len(df_clean)*100:.1f}%)")

# Visualizar top meanings
plt.figure(figsize=(14, 8))
meaning_counts.head(8).plot(kind='barh', color='lightblue')
plt.title('Principais Significados da Audição para os Respondentes', fontsize=14, fontweight='bold')
plt.xlabel('Número de Menções')
plt.tight_layout()
plt.show()

# 7. ANÁLISE DE RECURSOS DESEJADOS NO APP
print(f"\n📱 7. ANÁLISE DE RECURSOS DESEJADOS NO APP")

# Extrair features mais desejadas
app_features = []
for features in df_clean['desired_features']:
    if pd.notna(features):
        feature_list = [feature.strip() for feature in features.split(',')]
        app_features.extend(feature_list)

feature_counts = pd.Series(app_features).value_counts()
print("Top 10 recursos mais desejados:")
for i, (feature, count) in enumerate(feature_counts.head(10).items(), 1):
    print(f"{i:2d}. {feature}: {count} ({count/len(df_clean)*100:.1f}%)")

# Visualizar features
plt.figure(figsize=(14, 8))
feature_counts.head(10).plot(kind='barh', color='lightgreen')
plt.title('Recursos Mais Desejados no App de Audição', fontsize=14, fontweight='bold')
plt.xlabel('Número de Menções')
plt.tight_layout()
plt.show()

# 8. ANÁLISE DE DISPOSIÇÃO PARA PAGAMENTO
print(f"\n💰 8. ANÁLISE DE DISPOSIÇÃO PARA PAGAMENTO")

payment_willingness = df_clean['paid_app_interest'].value_counts()
print("Disposição para pagar por app:")
for option, count in payment_willingness.items():
    print(f"• {option}: {count} ({count/len(df_clean)*100:.1f}%)")

plt.figure(figsize=(10, 6))
payment_willingness.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Disposição para Pagar por App de Teste Auditivo', fontsize=14, fontweight='bold')
plt.ylabel('')
plt.show()

# 9. RESUMO ESTATÍSTICO
print(f"\n📊 9. RESUMO ESTATÍSTICO FINAL")

summary_stats = {
    'Total Respondentes': len(df_clean),
    'Faixa Etária Predominante': df_clean['age_group'].mode().iloc[0],
    'Interesse Alto em App (%)': len(df_clean[df_clean['app_interest'] == 'Yes, that would be helpful'])/len(df_clean)*100,
    'Nunca Fizeram Teste (%)': len(df_clean[df_clean['last_test_method'] == "I've never taken a hearing test"])/len(df_clean)*100,
    'Principal Barreira': df_clean['test_barrier'].mode().iloc[0],
    'Uso Fones >4h (%)': len(df_clean[df_clean['headphone_use'] == 'More than 4 hours'])/len(df_clean)*100,
    'Com Desconforto (%)': len(df_clean[df_clean['ear_discomfort'].isin(['Yes', 'Occasionally'])])/len(df_clean)*100
}

for metric, value in summary_stats.items():
    if isinstance(value, float):
        print(f"• {metric}: {value:.1f}")
    else:
        print(f"• {metric}: {value}")

print("="*80)
print("PARTE 2 CONCLUÍDA!")
print("="*80)

# %%
