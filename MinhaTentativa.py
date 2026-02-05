from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

print("Aprendendo a usar  embendding")

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


# TESTE
resultados = buscar_rea("historia")

# Verifica o tipo de retorno
if isinstance(resultados, dict):
    # Se for dicionário, pega a chave 'results'
    items = resultados.get("results", [])
elif isinstance(resultados, list):
    # Se já for lista, usa diretamente
    items = resultados
else:
    items = []

for i, r in enumerate(items, 1):
    print(f"{i}. {r.get('name')}")



modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Processando aulas: ")
print(f"\nQuantidade de aulas disponíveis:{len(items)}")
textos_para_embedding = []
for aula in items:
    # Combina o nome e descrição (se houver) para melhor busca
    texto = aula.get('name', '')
    textos_para_embedding.append(texto)

# Gera os embeddings com os textos
embenddings = modelo.encode(textos_para_embedding)
materiaUsuário = input()
def buscar_materia(consulta_usuario,topn=3,similaridade_minima=0.3):
    embenddings_consulta =modelo.encode([consulta_usuario])
    similaridades= cosine_similarity(embenddings_consulta,embenddings)[0]
    indices_ordenados=np.argsort(similaridades)[::-1]
    resultados=[]
    for indx in indices_ordenados:
        similaridade =similaridades[indx]
        if similaridade>=similaridade_minima:
            resultado={
                **items[indx],
                "relevancia":float(similaridade)
            }
            resultados.append(resultado)
            if len(resultados)>=topn:
                break        
    return resultados

print(f"\nConsulta: {materiaUsuário}")
print("\nResultados: ")
resultadoConsulta= buscar_materia(materiaUsuário)
if resultadoConsulta :
    for i,aula in enumerate(resultadoConsulta,1):
        print(f"\n{i},[{aula['relevancia']:.2%}]{aula['name']}")
        print(f"ID:{aula['id']}")
       
else:
    print ("Nenhuma aula encontrada")
