"""
Tutorial: Aprendendo sobre Embeddings
========================================

O que são Embeddings?
--------------------
Embeddings são representações numéricas (vetores) de dados como texto, imagens ou áudio.
Eles capturam o significado semântico e permitem que computadores "entendam" similaridade.

Exemplo: "gato" e "cachorro" terão embeddings similares (ambos são animais),
mas "gato" e "carro" terão embeddings muito diferentes.
"""

# ============================================================================
# PARTE 1: Embeddings Básicos com Sentence Transformers
# ============================================================================

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 70)
print("PARTE 1: Criando Embeddings de Texto")
print("=" * 70)

# Carregar modelo pré-treinado (multilíngue - funciona com português!)
modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Frases de exemplo
frases = [
    "O gato está dormindo no sofá",
    "Um felino descansa no móvel",
    "O carro está na garagem",
    "Python é uma linguagem de programação",
    "Eu amo programar em Python"
]

# Gerar embeddings (vetores numéricos)
embeddings = modelo.encode(frases)

print(f"\nNúmero de frases: {len(frases)}")
print(f"Dimensão de cada embedding: {embeddings[0].shape}")
print(f"\nPrimeiro embedding (primeiros 10 valores):")
print(embeddings[0][:10])

# ============================================================================
# PARTE 2: Calculando Similaridade
# ============================================================================

print("\n" + "=" * 70)
print("PARTE 2: Calculando Similaridade entre Frases")
print("=" * 70)

# Calcular similaridade entre todas as frases
similaridades = cosine_similarity(embeddings)

print("\nMatriz de Similaridade (0 a 1, onde 1 = idêntico):")
print("-" * 70)
for i, frase1 in enumerate(frases):
    print(f"\n'{frase1}':")
    for j, frase2 in enumerate(frases):
        if i != j:
            sim = similaridades[i][j]
            print(f"  vs '{frase2}': {sim:.4f}")

# ============================================================================
# PARTE 3: Busca Semântica
# ============================================================================

print("\n" + "=" * 70)
print("PARTE 3: Busca Semântica")
print("=" * 70)

# Base de conhecimento
documentos = [
    "Machine learning é um subcampo da inteligência artificial",
    "Deep learning usa redes neurais profundas",
    "Python é excelente para ciência de dados",
    "O Brasil é o maior país da América do Sul",
    "Futebol é o esporte mais popular no Brasil",
    "Embeddings transformam texto em vetores numéricos"
]

# Criar embeddings dos documentos
doc_embeddings = modelo.encode(documentos)

# Consulta do usuário
consulta = "Como funciona inteligência artificial?"
consulta_embedding = modelo.encode([consulta])

# Encontrar documentos mais similares
similaridades_consulta = cosine_similarity(consulta_embedding, doc_embeddings)[0]

# Ordenar por similaridade
indices_ordenados = np.argsort(similaridades_consulta)[::-1]

print(f"\nConsulta: '{consulta}'")
print("\nDocumentos mais relevantes:")
print("-" * 70)
for i, idx in enumerate(indices_ordenados[:3], 1):
    print(f"{i}. [{similaridades_consulta[idx]:.4f}] {documentos[idx]}")

# ============================================================================
# PARTE 4: Embeddings Personalizados (Exemplo Simples)
# ============================================================================

print("\n" + "=" * 70)
print("PARTE 4: Entendendo Embeddings - Visualização Simplificada")
print("=" * 70)

# Exemplo didático: embeddings 2D para visualização
palavras_exemplo = ["rei", "rainha", "homem", "mulher"]
print("\nEmbeddings 2D simplificados (apenas para demonstração):")
print("Em produção, embeddings têm centenas de dimensões!\n")

# Embeddings fictícios 2D para ilustrar conceito
embeddings_2d = {
    "rei": [0.9, 0.9],      # alto poder, masculino
    "rainha": [0.9, 0.1],   # alto poder, feminino
    "homem": [0.1, 0.9],    # baixo poder, masculino
    "mulher": [0.1, 0.1]    # baixo poder, feminino
}

for palavra, vetor in embeddings_2d.items():
    print(f"{palavra:10s}: {vetor}")

print("\nRelação vetorial famosa:")
print("rei - homem + mulher ≈ rainha")
resultado = np.array(embeddings_2d["rei"]) - np.array(embeddings_2d["homem"]) + np.array(embeddings_2d["mulher"])
print(f"Resultado calculado: {resultado}")
print(f"Embedding de rainha: {embeddings_2d['rainha']}")

# ============================================================================
# PARTE 5: Casos de Uso Práticos
# ============================================================================

print("\n" + "=" * 70)
print("PARTE 5: Aplicações Práticas de Embeddings")
print("=" * 70)

aplicacoes = """
1. BUSCA SEMÂNTICA
   - Encontrar documentos relevantes mesmo sem palavras exatas
   - Melhor que busca por palavras-chave tradicional

2. SISTEMAS DE RECOMENDAÇÃO
   - Recomendar produtos/conteúdos similares
   - Netflix, Spotify, Amazon usam embeddings

3. CHATBOTS E ASSISTENTES
   - Entender intenção do usuário
   - Responder perguntas semanticamente

4. CLASSIFICAÇÃO DE TEXTO
   - Categorizar emails, tickets, comentários
   - Análise de sentimento

5. DETECÇÃO DE DUPLICATAS
   - Encontrar textos similares
   - Combate a plágio

6. TRADUÇÃO AUTOMÁTICA
   - Mapear significados entre idiomas
   - Google Translate usa embeddings

7. RAG (Retrieval Augmented Generation)
   - Combinar embeddings com LLMs
   - ChatGPT com sua própria base de conhecimento
"""

print(aplicacoes)

# ============================================================================
# PRÓXIMOS PASSOS
# ============================================================================

print("=" * 70)
print("PRÓXIMOS PASSOS PARA CONTINUAR APRENDENDO")
print("=" * 70)

proximos_passos = """
1. Experimentar outros modelos:
   - all-MiniLM-L6-v2 (inglês, rápido)
   - all-mpnet-base-v2 (inglês, melhor qualidade)
   - neuralmind/bert-base-portuguese-cased (português)

2. Criar um sistema RAG:
   - Carregar seus próprios documentos
   - Criar base vetorial com FAISS ou ChromaDB
   - Integrar com OpenAI/Gemini

3. Fine-tuning:
   - Treinar embeddings para seu domínio específico
   - Melhorar precisão para seu caso de uso

4. Explorar outras bibliotecas:
   - OpenAI Embeddings (text-embedding-3-small)
   - Hugging Face Transformers
   - Cohere Embed

5. Visualizar embeddings:
   - Usar t-SNE ou UMAP para reduzir dimensionalidade
   - Plotar em 2D/3D com matplotlib

Para instalar dependências:
pip install sentence-transformers scikit-learn numpy
"""

print(proximos_passos)

print("\n" + "=" * 70)
print("Tutorial concluído! Execute este arquivo para ver os exemplos.")
print("=" * 70)
