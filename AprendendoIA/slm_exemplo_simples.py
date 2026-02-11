"""
SLM - Exemplo Simples e Prático
================================
Treinamento rápido de um Small Language Model para classificação
"""

print("🤖 Exemplo Simples de SLM - Classificação de Texto\n")

# Instalação necessária (descomente se precisar):
# !pip install transformers datasets torch scikit-learn

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import os

# ============================================================================
# PASSO 1: Criar dados de treinamento
# ============================================================================

print("📊 PASSO 1: Preparando dados de treinamento\n")

# Dados de exemplo - Classificação de tópicos
dados = {
    'texto': [
        # Tecnologia
        "Python é uma linguagem de programação muito popular",
        "JavaScript é essencial para desenvolvimento web",
        "Machine learning está revolucionando a indústria",
        "A inteligência artificial está em todo lugar",
        
        # Esportes
        "O time marcou três gols na partida de ontem",
        "O jogador fez um gol incrível de falta",
        "O campeonato começa na próxima semana",
        "A equipe treinou muito para a final",
        
        # Comida
        "Essa pizza estava deliciosa",
        "A receita do bolo é muito fácil",
        "Adoro massas italianas",
        "O restaurante serve comida excelente",
    ],
    'categoria': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    # 0 = Tecnologia, 1 = Esportes, 2 = Comida
}

categorias_nomes = {0: "Tecnologia", 1: "Esportes", 2: "Comida"}

dataset = Dataset.from_dict(dados)
print(f"✓ Dataset criado: {len(dataset)} exemplos")
print(f"✓ Categorias: {list(categorias_nomes.values())}\n")

# ============================================================================
# PASSO 2: Carregar modelo pequeno (com treinamento incremental)
# ============================================================================

print("🔧 PASSO 2: Carregando modelo\n")

# Verificar se já existe um modelo treinado anteriormente
CAMINHO_MODELO_SALVO = "./meu_slm_categorias"

if os.path.exists(CAMINHO_MODELO_SALVO):
    print("📂 Modelo anterior encontrado! Continuando treinamento...")
    print("   (Isso vai MELHORAR o modelo existente)\n")
    modelo_nome = CAMINHO_MODELO_SALVO
    eh_continuacao = True
else:
    print("🆕 Primeiro treinamento! Usando modelo base...\n")
    modelo_nome = "distilbert-base-multilingual-cased"
    eh_continuacao = False

tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
modelo = AutoModelForSequenceClassification.from_pretrained(
    modelo_nome,
    num_labels=3  # 3 categorias
)

if eh_continuacao:
    print(f"✓ Modelo: {CAMINHO_MODELO_SALVO} (Treinamento Contínuo)")
else:
    print(f"✓ Modelo: {modelo_nome} (Novo)")
print(f"✓ Parâmetros: {modelo.num_parameters():,}\n")

# ============================================================================
# PASSO 3: Preparar dados
# ============================================================================

print("⚙️ PASSO 3: Tokenizando dados\n")

def tokenizar(batch):
    return tokenizer(batch['texto'], padding='max_length', truncation=True, max_length=64)

dataset_preparado = dataset.map(tokenizar, batched=True)
dataset_preparado = dataset_preparado.rename_column("categoria", "labels")

print("✓ Dados tokenizados e prontos!\n")

# ============================================================================
# PASSO 4: Treinar
# ============================================================================

print("🚀 PASSO 4: Treinando modelo\n")

argumentos = TrainingArguments(
    output_dir="./modelo_categorias",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=3e-5,
    logging_steps=2,
    save_total_limit=1,
)

treinador = Trainer(
    model=modelo,
    args=argumentos,
    train_dataset=dataset_preparado,
)

print("Iniciando treinamento (pode demorar 1-2 minutos)...\n")
treinador.train()
print("\n✓ Treinamento concluído!\n")

# ============================================================================
# PASSO 5: Testar
# ============================================================================

print("🧪 PASSO 5: Testando modelo treinado\n")

# Criar pipeline para facilitar as predições
classificador = pipeline(
    "text-classification",
    model=modelo,
    tokenizer=tokenizer
)

# Textos de teste
testes = [
    "Aprendi a programar em Python hoje",
    "O atacante fez um hat-trick",
    "Essa lasanha estava perfeita",
    "Deep learning é fascinante",
    "O time ganhou de 2 a 0"
]

print("Resultados das predições:")
print("-" * 60)

for texto in testes:
    resultado = classificador(texto)[0]
    label_num = int(resultado['label'].split('_')[1])
    categoria = categorias_nomes[label_num]
    confianca = resultado['score']
    
    print(f"\n📝 Texto: {texto}")
    print(f"✓ Categoria: {categoria} (Confiança: {confianca:.1%})")

# ============================================================================
# BONUS: Salvar o modelo
# ============================================================================

print("\n" + "=" * 60)
print("💾 Salvando modelo...")

modelo.save_pretrained(CAMINHO_MODELO_SALVO)
tokenizer.save_pretrained(CAMINHO_MODELO_SALVO)

if eh_continuacao:
    print(f"✓ Modelo ATUALIZADO e salvo em: {CAMINHO_MODELO_SALVO}")
    print("  (Na próxima execução, vai continuar melhorando!)")
else:
    print(f"✓ Modelo NOVO salvo em: {CAMINHO_MODELO_SALVO}")
    print("  (Na próxima execução, vai usar este como base!)")
print("\n" + "=" * 60)
print("✨ Exemplo concluído com sucesso!")
print("=" * 60)
