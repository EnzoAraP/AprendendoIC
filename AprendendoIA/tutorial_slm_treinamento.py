"""
Tutorial: Small Language Models (SLM) - Treinamento
====================================================

O que são Small Language Models (SLM)?
---------------------------------------
SLMs são modelos de linguagem menores e mais eficientes que grandes modelos (LLMs).
- Mais rápidos para treinar e executar
- Requerem menos recursos computacionais
- Ideais para tarefas específicas
- Podem rodar em hardware comum

Neste tutorial, vamos aprender a:
1. Preparar dados para treinamento
2. Fazer fine-tuning de um modelo pequeno
3. Treinar um modelo do zero (básico)
4. Avaliar o modelo
"""

# ============================================================================
# PARTE 1: Fine-Tuning de um SLM Pré-Treinado
# ============================================================================

print("=" * 70)
print("PARTE 1: Fine-Tuning de um Small Language Model")
print("=" * 70)

# Bibliotecas necessárias (instale com: pip install transformers datasets torch)
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import torch
import numpy as np

# Dados de exemplo para classificação de sentimentos
dados_treinamento = {
    'texto': [
        "Eu adorei este produto! Muito bom!",
        "Péssima qualidade, não recomendo",
        "Excelente atendimento, super satisfeito",
        "Horrível, desperdício de dinheiro",
        "Produto mediano, nada excepcional",
        "Fantástico! Superou minhas expectativas",
        "Muito ruim, não funciona direito",
        "Bom custo-benefício, vale a pena",
        "Decepcionante, esperava mais",
        "Perfeito! Exatamente o que eu queria"
    ],
    'sentimento': [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  # 1=positivo, 0=negativo
}

# Criar dataset
dataset = Dataset.from_dict(dados_treinamento)

print(f"\nDataset criado com {len(dataset)} exemplos")
print(f"Primeiro exemplo: {dataset[0]}")

# ============================================================================
# PARTE 2: Preparar o Modelo e Tokenizador
# ============================================================================

print("\n" + "=" * 70)
print("PARTE 2: Configurando Modelo e Tokenizador")
print("=" * 70)

# Usar um modelo pequeno e eficiente
nome_modelo = "distilbert-base-multilingual-cased"  # ~135M parâmetros (pequeno!)

print(f"\nCarregando modelo: {nome_modelo}")
tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
modelo = AutoModelForSequenceClassification.from_pretrained(
    nome_modelo,
    num_labels=2  # 2 classes: positivo/negativo
)

print(f"Modelo carregado! Parâmetros: {modelo.num_parameters():,}")

# Função para tokenizar os textos
def tokenizar(exemplos):
    return tokenizer(
        exemplos['texto'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

# Tokenizar o dataset
dataset_tokenizado = dataset.map(tokenizar, batched=True)
dataset_tokenizado = dataset_tokenizado.rename_column("sentimento", "labels")

print("\nDados tokenizados e prontos para treino!")

# ============================================================================
# PARTE 3: Configurar e Executar o Treinamento
# ============================================================================

print("\n" + "=" * 70)
print("PARTE 3: Treinamento do Modelo")
print("=" * 70)

# Configurações de treinamento
args_treinamento = TrainingArguments(
    output_dir="./modelo_treinado",
    num_train_epochs=3,              # Número de épocas
    per_device_train_batch_size=2,   # Tamanho do batch (pequeno para rodar em qualquer PC)
    learning_rate=2e-5,               # Taxa de aprendizado
    logging_steps=5,
    save_steps=10,
    eval_strategy="no",               # Sem validação neste exemplo simples
    save_total_limit=1,
)

# Criar o treinador
treinador = Trainer(
    model=modelo,
    args=args_treinamento,
    train_dataset=dataset_tokenizado,
)

print("\nIniciando treinamento...")
print("(Isso pode levar alguns minutos dependendo do seu hardware)")

# Treinar o modelo
resultado = treinador.train()

print(f"\nTreinamento concluído!")
print(f"Loss final: {resultado.training_loss:.4f}")

# ============================================================================
# PARTE 4: Testando o Modelo Treinado
# ============================================================================

print("\n" + "=" * 70)
print("PARTE 4: Testando o Modelo")
print("=" * 70)

def prever_sentimento(texto):
    """Função para prever o sentimento de um novo texto"""
    inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = modelo(**inputs)
    
    probabilidades = torch.softmax(outputs.logits, dim=1)
    predicao = torch.argmax(probabilidades, dim=1).item()
    confianca = probabilidades[0][predicao].item()
    
    sentimento = "POSITIVO 😊" if predicao == 1 else "NEGATIVO 😞"
    return sentimento, confianca

# Testar com novos textos
textos_teste = [
    "Este produto é maravilhoso!",
    "Não gostei, muito caro",
    "Recomendo para todos, excelente qualidade"
]

print("\nTestando o modelo com novos textos:")
print("-" * 70)

for texto in textos_teste:
    sentimento, confianca = prever_sentimento(texto)
    print(f"\nTexto: {texto}")
    print(f"Sentimento: {sentimento} (Confiança: {confianca:.2%})")

# ============================================================================
# PARTE 5: Salvando e Carregando o Modelo
# ============================================================================

print("\n" + "=" * 70)
print("PARTE 5: Salvando o Modelo")
print("=" * 70)

# Salvar modelo e tokenizador
caminho_salvar = "./meu_slm_sentimentos"
modelo.save_pretrained(caminho_salvar)
tokenizer.save_pretrained(caminho_salvar)

print(f"\nModelo salvo em: {caminho_salvar}")
print("\nPara carregar depois, use:")
print(f"modelo = AutoModelForSequenceClassification.from_pretrained('{caminho_salvar}')")
print(f"tokenizer = AutoTokenizer.from_pretrained('{caminho_salvar}')")

# ============================================================================
# CONCEITOS IMPORTANTES
# ============================================================================

print("\n" + "=" * 70)
print("CONCEITOS IMPORTANTES SOBRE TREINAMENTO DE SLM")
print("=" * 70)

conceitos = """
1. FINE-TUNING vs TREINAR DO ZERO:
   - Fine-tuning: Ajustar um modelo pré-treinado (mais rápido e eficiente)
   - Do zero: Treinar completamente novo (requer MUITO mais dados e tempo)

2. HIPERPARÂMETROS PRINCIPAIS:
   - learning_rate: Velocidade de aprendizado (2e-5 a 5e-5 é típico)
   - batch_size: Quantos exemplos processar por vez
   - num_epochs: Quantas vezes passar por todo o dataset
   
3. OVERFITTING:
   - Quando o modelo "decora" os dados de treino
   - Solução: Mais dados, regularização, early stopping
   
4. TIPOS DE SLM:
   - DistilBERT: ~65M parâmetros (versão destilada do BERT)
   - TinyBERT: ~14M parâmetros (ainda menor)
   - MobileBERT: Otimizado para dispositivos móveis
   
5. DATASETS:
   - Mínimo: ~1000 exemplos para fine-tuning
   - Ideal: 10.000+ para bons resultados
   - Qualidade > Quantidade
"""

print(conceitos)

# ============================================================================
# PRÓXIMOS PASSOS
# ============================================================================

print("\n" + "=" * 70)
print("PRÓXIMOS PASSOS PARA APRENDER MAIS")
print("=" * 70)

proximos_passos = """
1. Experimente com mais dados
2. Teste diferentes modelos (TinyBERT, MiniLM)
3. Implemente validação e métricas de avaliação
4. Aprenda sobre data augmentation
5. Explore outras tarefas: NER, Q&A, geração de texto
6. Estude técnicas de otimização (quantização, pruning)

RECURSOS ÚTEIS:
- Hugging Face Course: https://huggingface.co/course
- Datasets Hub: https://huggingface.co/datasets
- Model Hub: https://huggingface.co/models
"""

print(proximos_passos)

print("\n" + "=" * 70)
print("Tutorial concluído! 🎉")
print("=" * 70)
