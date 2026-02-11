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
        "limit":30
    }
    response = requests.get(url,params = params)
    response.raise_for_status()
    return response.json()
def buscarReaEduplay(assunto):
    url = "https://eduplay.rnp.br/api/v1/search"
    params = {
         "term":assunto,
         "quantity":30,
         "page":1,
         "type":0,
         "order":0
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def buscarReaAquarela(assunto):
    return
    
def retrieve (query:str, top_k:int=5):
    modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    dadosTodosTexto = []
    dadosTodosDicionario = []
    resultadoMEdcred = buscarReaMedcred(query)
    if isinstance(resultadoMEdcred, dict):
     dadosMedcredDicionario = resultadoMEdcred.get("results", [])
    elif isinstance(resultadoMEdcred, list):
     dadosMedcredDicionario = resultadoMEdcred
    else:
     dadosMedcredDicionario = []

    resultadoEduplay = buscarReaEduplay(query)
    if isinstance(resultadoEduplay, dict):
     dadosEduplayDicionario = resultadoEduplay.get("contents", [])
    for item in dadosMedcredDicionario:
        item['fonte'] ='medcred'
        dadosTodosDicionario.append(item)
    for item in dadosEduplayDicionario:
        item['fonte'] ='eduplay'
        dadosTodosDicionario.append(item)
 
    for dados in dadosMedcredDicionario:
        nomematerial = dados.get('name',[])
        dadosTodosTexto.append(nomematerial)
    for dados in dadosEduplayDicionario:
        nomematerial = dados.get('name',[])
        dadosTodosTexto.append(nomematerial)

    embbendingdadosREAs = modelo.encode(dadosTodosTexto)
    embbendingConsulta = modelo.encode([query])
    
    similaridades = cosine_similarity(embbendingConsulta,embbendingdadosREAs)[0]
    indices_ordenados = np.argsort(similaridades)[::-1]

    topKtotal = []
 
    for indx in indices_ordenados:
        similaridade = similaridades[indx]
        resultado = {
        **dadosTodosDicionario[indx],
        "relevancia":float(similaridade)
        }
        topKtotal.append(resultado)
        
        if len(topKtotal)>=top_k:
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

def formatacaoDadosEduplay(listadados):
    frasefinal =[]
    for dados in listadados:
        nomeMaterial = dados.get('name','Sem título')
        descricao = dados.get('description','Sem descricao')
        contenttype = dados.get('contentType',"Desconhecido")
        userOwner = dados.get('userOwner',{}).get('name','Desconhecido')
        link = dados.get('embedUrl','Não possui')
        fraseMomentanea = f"Título do material: {nomeMaterial}, descrição do material: {descricao},  tipo de material: {contenttype}, dono do material: {userOwner}, link do material: {link} \n"
        frasefinal.append(fraseMomentanea)
    return "\n".join(frasefinal)

def formatacaotodos(listadados):
   
    frasefinal = []
    for dados in listadados:

        if dados.get('fonte') == 'medcred':
            usuarioNome = dados.get('user',{}).get('name','Desconhecido')
            nomematerial = dados.get('name','Sem título')
            views = dados.get('views',"Não descrito")
            likes = dados.get('likes',"Não descrito")
            relevancia = dados.get('relevancia',"Erro")
            fraseMomentanea = f" Título do material: {nomematerial}, views do material: {views}, likes do material: {likes}, dono do material: {usuarioNome}, relevância em relação ao tema: {relevancia} \n"
            frasefinal.append(fraseMomentanea)
        if dados.get('fonte') == 'eduplay': 
            nomeMaterial = dados.get('name','Sem título')
            descricao = dados.get('description','Sem descricao')
            contenttype = dados.get('contentType',"Desconhecido")
            userOwner = dados.get('userOwner',{}).get('name','Desconhecido')
            link = dados.get('embedUrl','Não possui')
            relevancia = dados.get('relevancia',"Erro")
            fraseMomentanea = f"Título do material: {nomeMaterial}, descrição do material: {descricao},  tipo de material: {contenttype}, dono do material: {userOwner}, link do material: {link},relavância: {relevancia}\n"
            frasefinal.append(fraseMomentanea)
    return "\n".join(frasefinal)
pedido =input()

resultadoRetrieve = retrieve(pedido,5)

print(f"Resultado final: \n {formatacaotodos(resultadoRetrieve)}")


    