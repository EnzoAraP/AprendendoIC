from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HF_TOKEN")
def buscarReaMedcred(assunto):
    url = "https://api.mecred.c3sl.ufpr.br/public/elastic/search"
    params = {
        "indexes":"resources",
        "query":assunto,
        "limit":20
    }
    response = requests.get(url,params = params)
    response.raise_for_status()
    return response.json()
    
def retrieve (query:str, top_k:int=5):
    modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    dadosMedcredTexto= []
    dadosTodosTexto = []
    dadosTodosDicionario = []
    resultadoMEdcred = buscarReaMedcred(query)
    if isinstance(resultadoMEdcred, dict):
     dadosMedcredDicionario = resultadoMEdcred.get("results", [])
    elif isinstance(resultadoMEdcred, list):
  
     dadosMedcredDicionario = resultadoMEdcred
    else:
     dadosMedcredDicionario = []
    print(f"Resultado sem embbending{formatacaoDadosMedCred(dadosMedcredDicionario)}")
    dadosTodosDicionario.append(dadosMedcredDicionario)
    print(f"tamanhodadosmedcredDIcionario{len(dadosMedcredDicionario)}")
    for dados in dadosMedcredDicionario:
        nomematerialMed = dados.get('name',[])
        dadosMedcredTexto.append(nomematerialMed)
        dadosTodosTexto.append(dadosMedcredTexto)
    print(f"tamanho medcredtexto{len(dadosMedcredTexto)}")
    embbendingdadosREAs = modelo.encode(dadosMedcredTexto)
    embbendingConsulta = modelo.encode([query])
    
    similaridades = cosine_similarity(embbendingConsulta,embbendingdadosREAs)[0]
    indices_ordenados = np.argsort(similaridades)[::-1]
    print(f"tamanho indice ord {len(indices_ordenados)}")
    topKtotal = []
    print(len(topKtotal))
    for indx in indices_ordenados:
        similaridade = similaridades[indx]
        resultado = {
        **dadosMedcredDicionario[indx],
        "relevancia":float(similaridade)
        }
        topKtotal.append(resultado)
        print(len(topKtotal))
        if len(topKtotal)>=top_k:
            print(f"entrou break{len(topKtotal)}")
            break
    return topKtotal

def formatacaoDadosMedCred(listadados):
    frasefinal =[]
    for dados in listadados:
        usuarioNome = dados.get('user',{}).get('name','Desconhecido')
        nomematerial = dados.get('name','Sem título')
        views = dados.get('views',"Não descrito")
        likes = dados.get('likes',"Não descrito")
        relevancia = dados.get('relevancia',"Erro")
        fraseMomentanea = f" Título do material: {nomematerial}, views do material: {views}, likes do material: {likes}, dono do material: {usuarioNome}, relevância em relação ao tema: {relevancia} \n"
        frasefinal.append(fraseMomentanea)
    return "\n".join (frasefinal)


pedido =input()
resultadoRetrieve = retrieve(pedido,5)
print(f"Resultado final: {formatacaoDadosMedCred(resultadoRetrieve)}")


    