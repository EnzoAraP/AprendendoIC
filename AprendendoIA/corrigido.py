from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

print("Aprendendo a usar embendding")
print("Carregando modelo... (isso pode demorar na primeira vez)")

# Carrega o modelo UMA VEZ no início
modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def buscar_rea(assunto):
    url = "https://api.mecred.c3sl.ufpr.br/public/elastic/search"
    params = {
        "indexes": "resources",
        "query": assunto,
        "limit": 40
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

# Pergunta primeiro o que o usuário quer buscar
print("Qual matéria você quer pesquisar?")
materiaUsuário = input()

# Busca na API baseado no que o usuário digitou
print(f"\nBuscando '{materiaUsuário}' na API...")
items = buscar_rea(materiaUsuário)

print(f"Encontrados {len(items)} resultados")
print("Processando com embeddings...")

# Prepara os textos para embeddings
textos_para_embedding = []
for aula in items:
    texto = aula.get('name', '')
    if aula.get('description'):
        texto += ' ' + aula.get('description', '')
    textos_para_embedding.append(texto)

# Gera os embeddings
embenddings = modelo.encode(textos_para_embedding)

def buscar_materia(consulta_usuario, topn=5, similaridade_minima=0.3):
    embenddings_consulta = modelo.encode([consulta_usuario])
    similaridades = cosine_similarity(embenddings_consulta, embenddings)[0]
    indices_ordenados = np.argsort(similaridades)[::-1]
    resultados = []
    
    for indx in indices_ordenados:
        similaridade = similaridades[indx]
        if similaridade >= similaridade_minima:
            resultado = {
                **items[indx],
                "relevancia": float(similaridade)
            }
            resultados.append(resultado)
            if len(resultados) >= topn:
                break        
    return resultados

print(f"\nConsulta: {materiaUsuário}")
print("\nResultados ordenados por relevância:")
resultadoConsulta = buscar_materia(materiaUsuário)

if resultadoConsulta:
    for i, aula in enumerate(resultadoConsulta, 1):
        print(f"\n{i}. [{aula['relevancia']:.2%}] {aula['name']}")
        print(f"   ID: {aula['id']}")
        print(f"   Views: {aula['views']} | Likes: {aula['likes']}")
else:
    print("Nenhuma aula encontrada com similaridade suficiente")