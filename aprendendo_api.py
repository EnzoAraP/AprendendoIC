"""
Tutorial: Aprendendo sobre APIs em Python
==========================================

O que é uma API?
----------------
API (Application Programming Interface) é uma forma de sistemas se comunicarem.
Pense como um "garçom" que leva seu pedido (requisição) para a cozinha (servidor)
e traz de volta a comida (resposta).

Exemplo: Quando você usa um app de clima, ele faz uma requisição para uma API
que retorna a temperatura, chuva, etc.
"""

import requests
import json
from datetime import datetime

# ============================================================================
# PARTE 1: Requisições GET - Buscar Dados
# ============================================================================

print("=" * 70)
print("PARTE 1: Fazendo sua Primeira Requisição GET")
print("=" * 70)

# API pública gratuita - JSONPlaceholder (API de teste)
url = "https://jsonplaceholder.typicode.com/users"

print(f"\n🌐 Fazendo requisição para: {url}")

# Fazer a requisição GET
resposta = requests.get(url)

# Verificar se deu certo (status code 200 = sucesso)
print(f"Status Code: {resposta.status_code}")

if resposta.status_code == 200:
    print("✓ Requisição bem-sucedida!")
    
    # Converter JSON para objeto Python
    usuarios = resposta.json()
    
    print(f"\n📊 Total de usuários retornados: {len(usuarios)}")
    print("\nPrimeiros 3 usuários:")
    print("-" * 70)
    
    for usuario in usuarios[:3]:
        print(f"\nID: {usuario['id']}")
        print(f"Nome: {usuario['name']}")
        print(f"Email: {usuario['email']}")
        print(f"Cidade: {usuario['address']['city']}")
else:
    print("❌ Erro na requisição!")

# ============================================================================
# PARTE 2: Parâmetros de Query - Filtrar Dados
# ============================================================================

print("\n" + "=" * 70)
print("PARTE 2: Usando Parâmetros de Query")
print("=" * 70)

# Buscar posts de um usuário específico
url_posts = "https://jsonplaceholder.typicode.com/posts"

# Parâmetros para filtrar (userId=1)
parametros = {
    "userId": 1
}

print(f"\n🔍 Buscando posts do usuário 1...")
resposta = requests.get(url_posts, params=parametros)

if resposta.status_code == 200:
    posts = resposta.json()
    print(f"✓ Encontrados {len(posts)} posts")
    
    print("\nPrimeiros 2 posts:")
    print("-" * 70)
    for post in posts[:2]:
        print(f"\nID: {post['id']}")
        print(f"Título: {post['title']}")
        print(f"Conteúdo: {post['body'][:50]}...")

# ============================================================================
# PARTE 3: Requisições POST - Enviar Dados
# ============================================================================

print("\n" + "=" * 70)
print("PARTE 3: Criando Dados com POST")
print("=" * 70)

# Dados para criar um novo post
novo_post = {
    "title": "Aprendendo APIs",
    "body": "Este é um post criado via API usando Python!",
    "userId": 1
}

print("\n📝 Criando novo post...")
print(f"Dados enviados: {novo_post}")

resposta = requests.post(url_posts, json=novo_post)

if resposta.status_code == 201:  # 201 = Created
    post_criado = resposta.json()
    print("\n✓ Post criado com sucesso!")
    print(f"ID do novo post: {post_criado['id']}")
    print(f"Título: {post_criado['title']}")

# ============================================================================
# PARTE 4: Headers e Autenticação
# ============================================================================

print("\n" + "=" * 70)
print("PARTE 4: Trabalhando com Headers")
print("=" * 70)

# Headers customizados (útil para autenticação, tipo de conteúdo, etc.)
headers = {
    "Content-Type": "application/json",
    "User-Agent": "MeuApp/1.0",
    # "Authorization": "Bearer SEU_TOKEN_AQUI"  # Exemplo de autenticação
}

print("\n📋 Enviando requisição com headers customizados...")
resposta = requests.get(url, headers=headers)

print(f"Status: {resposta.status_code}")
print(f"Content-Type da resposta: {resposta.headers.get('Content-Type')}")

# ============================================================================
# PARTE 5: API Real - ViaCEP (Buscar Endereço por CEP)
# ============================================================================

print("\n" + "=" * 70)
print("PARTE 5: Exemplo Prático - API ViaCEP")
print("=" * 70)

def buscar_cep(cep):
    """
    Busca informações de endereço pelo CEP usando a API ViaCEP
    
    Args:
        cep: CEP no formato "01310-100" ou "01310100"
    
    Returns:
        Dicionário com dados do endereço ou None se não encontrado
    """
    # Limpar formatação do CEP
    cep_limpo = cep.replace("-", "").replace(".", "")
    
    # URL da API ViaCEP
    url = f"https://viacep.com.br/ws/{cep_limpo}/json/"
    
    print(f"\n🔍 Buscando CEP: {cep}")
    
    try:
        resposta = requests.get(url, timeout=5)
        
        if resposta.status_code == 200:
            dados = resposta.json()
            
            # Verificar se CEP existe
            if "erro" not in dados:
                return dados
            else:
                print("❌ CEP não encontrado!")
                return None
        else:
            print(f"❌ Erro na requisição: {resposta.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Timeout - API demorou muito para responder")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro na requisição: {e}")
        return None

# Testar com CEPs reais
ceps_teste = ["01310-100", "20040-020", "30140-071"]

for cep in ceps_teste:
    resultado = buscar_cep(cep)
    
    if resultado:
        print("✓ Endereço encontrado:")
        print(f"   Logradouro: {resultado['logradouro']}")
        print(f"   Bairro: {resultado['bairro']}")
        print(f"   Cidade: {resultado['localidade']}/{resultado['uf']}")

# ============================================================================
# PARTE 6: Tratamento de Erros
# ============================================================================

print("\n" + "=" * 70)
print("PARTE 6: Tratamento de Erros")
print("=" * 70)

def requisicao_segura(url):
    """
    Faz requisição com tratamento completo de erros
    """
    try:
        print(f"\n🌐 Requisitando: {url}")
        
        # Timeout de 5 segundos
        resposta = requests.get(url, timeout=5)
        
        # Lançar exceção para status codes de erro (4xx, 5xx)
        resposta.raise_for_status()
        
        print(f"✓ Sucesso! Status: {resposta.status_code}")
        return resposta.json()
        
    except requests.exceptions.Timeout:
        print("❌ Erro: Timeout (servidor demorou muito)")
        return None
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ Erro HTTP: {e}")
        return None
        
    except requests.exceptions.ConnectionError:
        print("❌ Erro de conexão (sem internet?)")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro inesperado: {e}")
        return None
        
    except json.JSONDecodeError:
        print("❌ Erro ao decodificar JSON")
        return None

# Testar com URL válida
dados = requisicao_segura("https://jsonplaceholder.typicode.com/users/1")
if dados:
    print(f"Nome do usuário: {dados['name']}")

# Testar com URL inválida
dados = requisicao_segura("https://site-que-nao-existe-12345.com")

# ============================================================================
# PARTE 7: Integração com Embeddings (Exemplo Avançado)
# ============================================================================

print("\n" + "=" * 70)
print("PARTE 7: Combinando API + Embeddings")
print("=" * 70)

# Buscar posts da API
print("\n📚 Buscando posts da API...")
resposta = requests.get("https://jsonplaceholder.typicode.com/posts")

if resposta.status_code == 200:
    posts = resposta.json()[:10]  # Pegar só 10 posts
    
    print(f"✓ {len(posts)} posts baixados da API")
    
    # Opcional: Se quiser usar embeddings para buscar posts similares
    print("\n💡 Você poderia usar embeddings aqui para:")
    print("   - Criar embeddings de cada post")
    print("   - Buscar posts similares semanticamente")
    print("   - Agrupar posts por tópico")
    
    print("\n📝 Exemplo de posts baixados:")
    for i, post in enumerate(posts[:3], 1):
        print(f"\n{i}. {post['title']}")

# ============================================================================
# PARTE 8: APIs Populares para Praticar
# ============================================================================

print("\n" + "=" * 70)
print("PARTE 8: APIs Públicas para Praticar")
print("=" * 70)

apis_pratica = """
APIs GRATUITAS SEM AUTENTICAÇÃO:
---------------------------------
1. JSONPlaceholder
   - URL: https://jsonplaceholder.typicode.com
   - Uso: API de teste com posts, usuários, comentários

2. ViaCEP
   - URL: https://viacep.com.br
   - Uso: Buscar endereços por CEP (Brasil)

3. Dog API
   - URL: https://dog.ceo/api/breeds/image/random
   - Uso: Fotos aleatórias de cachorros

4. PokéAPI
   - URL: https://pokeapi.co/api/v2/pokemon/pikachu
   - Uso: Dados de Pokémon

5. RestCountries
   - URL: https://restcountries.com/v3.1/name/brazil
   - Uso: Informações sobre países

APIs QUE PRECISAM DE REGISTRO (GRATUITAS):
------------------------------------------
1. OpenWeatherMap
   - Clima em tempo real
   - https://openweathermap.org/api

2. NewsAPI
   - Notícias de várias fontes
   - https://newsapi.org

3. GitHub API
   - Dados de repositórios
   - https://api.github.com

4. OpenAI API
   - GPT, embeddings, etc.
   - https://platform.openai.com
"""

print(apis_pratica)

# ============================================================================
# EXERCÍCIOS PRÁTICOS
# ============================================================================

print("\n" + "=" * 70)
print("EXERCÍCIOS PARA PRATICAR")
print("=" * 70)

exercicios = """
1. BÁSICO: Buscar Pokémon
   - Use https://pokeapi.co/api/v2/pokemon/pikachu
   - Imprima o nome, altura e peso do Pokémon

2. INTERMEDIÁRIO: Sistema de Busca de CEP
   - Crie uma função que pede o CEP ao usuário
   - Busque na API ViaCEP
   - Mostre o endereço completo formatado

3. AVANÇADO: Comparar Posts com Embeddings
   - Baixe 20 posts do JSONPlaceholder
   - Crie embeddings dos títulos
   - Permita buscar posts similares

4. DESAFIO: Dashboard de Clima
   - Registre em OpenWeatherMap (grátis)
   - Crie um programa que mostra clima de várias cidades
   - Salve histórico em arquivo JSON

5. PROJETO: Sistema RAG com API
   - Baixe artigos de uma API
   - Crie embeddings
   - Permita fazer perguntas sobre os artigos
"""

print(exercicios)

# ============================================================================
# CÓDIGO DE EXEMPLO: Teste Rápido de API
# ============================================================================

print("\n" + "=" * 70)
print("TESTE RÁPIDO: Dog API")
print("=" * 70)

print("\n🐕 Buscando foto aleatória de cachorro...")
resposta = requests.get("https://dog.ceo/api/breeds/image/random")

if resposta.status_code == 200:
    dados = resposta.json()
    print("✓ Sucesso!")
    print(f"URL da foto: {dados['message']}")
    print("\n💡 Cole essa URL no navegador para ver a foto!")

print("\n" + "=" * 70)
print("Tutorial concluído! Agora você sabe usar APIs em Python!")
print("=" * 70)
print("\nPróximos passos:")
print("1. Pratique com as APIs sugeridas")
print("2. Combine APIs com embeddings")
print("3. Crie projetos reais (clima, notícias, etc.)")
