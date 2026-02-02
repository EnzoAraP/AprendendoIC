from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("Aprendendo a usar  embendding")

modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
aulasDisponiveis =[
    {
        "id":1,
        'titulo':"Introdução a Python",
        'descricao':"Aprenda os fundamentos básicos  para programação em python "  
    },  
    {
        "id":2,
        'titulo':"Desenvolvimento  web",
        'descricao':"Aula sobre desenvolvimento web com enfoque em HTML,CSS,JAVASCRIPT"  
    },    
    {
        "id":3,
        'titulo':" Minecraft  ensinando java",
        'descricao':"Aprenda  java com minecraft"  
    },  
    {
        "id":4,
        'titulo':"Amoungus ",
        'descricao':"SUS "  
    },
]
print("Processando aulas: ")
dadosAulas=[
    f"{aula['titulo']}. {aula['descricao']}" 
    for aula in aulasDisponiveis
    ]
embenddings = modelo.encode(dadosAulas)
print(f"\nQuantidade de aulas disponíveis:{len(aulasDisponiveis)}")
print(" Qual matéria vocÊ quer pesquisar sobre? ")
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
                **aulasDisponiveis[indx],
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
        print(f"\n{i},[{aula['relevancia']:.2%}]{aula['titulo']}")
        print(f"ID:{aula['id']}")
        print(f"{aula['descricao']}")
else:
    print ("Nenhuma aula encontrada")