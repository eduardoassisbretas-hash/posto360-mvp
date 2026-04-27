import base64
import json
import os
import re
import shutil
import threading
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import List

import requests
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ENV_PATH = os.path.join(PROJECT_DIR, ".env")
APP_ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)
load_dotenv(APP_ENV_PATH)

app = FastAPI()

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MAX_UPLOAD_IMAGES = 30
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1200
PROCESSING_LOCK = threading.Lock()
RETENTION_DAYS = 15
RECENT_ITEMS_LIMIT = 5
READINGS_PAGE_SIZE = 10
VISION_MAX_WORKERS = 4
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")
OPENAI_VISION_MIN_CONFIDENCE = float(os.getenv("OPENAI_VISION_MIN_CONFIDENCE", "0.20"))
OPENAI_VISION_LOW_CONFIDENCE = float(os.getenv("OPENAI_VISION_LOW_CONFIDENCE", "0.65"))
DECISION_MIN_VALID_READINGS = int(os.getenv("DECISION_MIN_VALID_READINGS", "2"))
ALERT_MIN_COMPETITOR_READINGS = int(os.getenv("ALERT_MIN_COMPETITOR_READINGS", "2"))
DEBUG_VISION_FOLDER = os.path.join(UPLOAD_FOLDER, "debug_vision")
OPENAI_CONFIG_ERROR_MESSAGE = (
    "OPENAI_API_KEY ausente. Configure a chave no arquivo .env na raiz do projeto."
)
COMBUSTIVEIS_DECISAO = {
    "gasolina_comum",
    "gasolina_aditivada",
    "gasolina_premium",
    "etanol",
    "diesel",
    "diesel_s10",
}
COMBUSTIVEIS_CORE_DECISAO = {
    "gasolina_comum",
    "etanol",
    "diesel",
    "diesel_s10",
}
PRICE_SANITY_RANGES = {
    "etanol": (2.0, 7.0),
    "gasolina": (3.5, 9.0),
    "diesel": (3.5, 10.0),
    "nao_identificado": (2.0, 10.0),
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEBUG_VISION_FOLDER, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)
reverse_geocode_cache = {}

if not OPENAI_API_KEY:
    print(f"[CONFIG] ERRO: {OPENAI_CONFIG_ERROR_MESSAGE}")

dados_dashboard = {
    "analises": [],
    "alertas": [],
    "precos_imagem": [],
    "leituras_imagem": [],
    "resumo": {},
    "resumo_por_tipo": {},
    "comparativos": [],
    "upload_regioes": {},
    "upload_postos": {},
    "upload_posto_ids": {},
    "upload_localizacoes": {},
    "upload_tipos": {},
    "postos_cadastrados": [],
    "configuracoes": {
        "perfil": {
            "nome": "",
            "email": "",
            "empresa": "",
        },
        "regioes": [],
        "parametros": {
            "margem_acima_media": "0.05",
            "margem_alerta_critico": "0.10",
            "estrategia": "Equilibrado",
        },
        "alertas_config": {
            "preco_acima_media": True,
            "concorrentes_reduziram": True,
            "oportunidade_margem": True,
        },
    },
    "processamento_upload": {
        "em_andamento": False,
        "total": 0,
        "processadas": 0,
    },
    "reset_token": 0,
}


def calcular_resumo(precos: List[float]) -> dict:
    if not precos:
        return {}

    return {
        "maior_preco": f"R$ {max(precos):.2f}",
        "menor_preco": f"R$ {min(precos):.2f}",
        "media_regiao": f"R$ {sum(precos) / len(precos):.2f}",
    }


def calcular_alertas_operacionais(analises: List[dict]) -> List[dict]:
    alertas = []

    for item in analises:
        if item["dias_restantes"] < 3:
            if not posto_informado(item.get("posto")):
                continue

            posto = (item.get("posto") or "").strip()
            regiao = (item.get("regiao") or "").strip() or "Região não informada"
            alertas.append(
                {
                    "titulo": "Crítico",
                    "classe_dashboard": "alert-card-critical",
                    "classe_lista": "critical",
                    "posto": posto,
                    "regiao": regiao,
                    "combustivel": item.get("produto") or "Combustível",
                    "diferenca": "Estoque abaixo do ideal",
                    "acao": "Pode perder volume",
                    "sugestao": "Sugestão: revisar abastecimento",
                    "mensagem": f"{posto} · {item.get('produto') or 'Combustível'} · Pode perder volume",
                }
            )

    return alertas


def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(char for char in texto if not unicodedata.combining(char))
    texto = re.sub(r"\bs[\s\-]*1[\s\-]*0\b", "s10", texto)
    texto = re.sub(r"\bs[\s\-]*1[\s\-]*o\b", "s10", texto)
    texto = re.sub(r"\bs[\s\-]*i[\s\-]*0\b", "s10", texto)
    texto = re.sub(r"\b5[\s\-]*1[\s\-]*0\b", "s10", texto)
    return re.sub(r"\s+", " ", texto).strip()


def detectar_tipo_combustivel(texto: str) -> str:
    tipo_visao = detectar_tipo_visao(texto)

    if tipo_visao:
        return tipo_visao

    return "gasolina_comum"


def detectar_tipo_visao(texto: str) -> str | None:
    texto_normalizado = normalizar_texto(texto)
    texto_com_espacos = f" {texto_normalizado} "

    if (
        "etanol" in texto_normalizado
        or "alcool" in texto_normalizado
        or "eta" in texto_normalizado
        or " e " in texto_com_espacos
        or texto_normalizado.startswith("e")
    ):
        return "etanol"

    if (
        "s10" in texto_normalizado
        or "s-10" in texto_normalizado
        or "s 10" in texto_normalizado
        or "diesel s10" in texto_normalizado
        or "diesel" in texto_normalizado
        or "dsl" in texto_normalizado
        or texto_normalizado.startswith("d ")
        or texto_normalizado == "d"
        or " d " in texto_com_espacos
        or texto_normalizado.startswith("d")
    ):
        return "diesel"

    if (
        "aditiv" in texto_normalizado
        or "vpower" in texto_normalizado
        or "v-power" in texto_normalizado
        or "premium" in texto_normalizado
    ):
        return "gasolina_aditivada"

    if (
        "gasolina" in texto_normalizado
        or "gas" in texto_normalizado
        or " g " in texto_com_espacos
        or "comum" in texto_normalizado
        or texto_normalizado.startswith("g")
    ):
        return "gasolina_comum"

    return None


def identificar_tipo_combustivel(texto: str) -> str:
    return detectar_tipo_combustivel(texto)


def detectar_tipo_linha(texto_linha: str) -> str | None:
    linha = normalizar_texto(texto_linha)
    linha_com_espacos = f" {linha} "

    marcadores_diesel = [
        "diesel",
        "s10",
        "s-10",
        "s 10",
        "510",
        "sio",
        "s1o",
    ]
    marcadores_etanol = [
        "etanol",
        "alcool",
        "etan0l",
        "etanql",
    ]
    marcadores_gasolina = [
        "gasolina",
        "gas",
        "comum",
    ]

    if any(item in linha for item in marcadores_diesel) or " d " in linha_com_espacos or linha.startswith("d"):
        return "diesel"

    if any(item in linha for item in marcadores_etanol) or " e " in linha_com_espacos or linha.startswith("e"):
        return "etanol"

    if any(item in linha for item in marcadores_gasolina) or " g " in linha_com_espacos or linha.startswith("g"):
        return "gasolina_comum"

    return None


def linha_prioritaria_diesel(linha: str, preco: float | None) -> bool:
    if preco is None:
        return False

    linha_normalizada = normalizar_texto(linha)
    linha_com_espacos = f" {linha_normalizada} "

    return (
        "diesel s10" in linha_normalizada
        or "diesel" in linha_normalizada
        or "s10" in linha_normalizada
        or "s-10" in linha_normalizada
        or "s 10" in linha_normalizada
        or linha_normalizada == "d"
        or linha_normalizada.startswith("d ")
        or " d " in linha_com_espacos
    )


def preco_dentro_da_faixa(tipo: str | None, preco: float | None) -> bool:
    if tipo is None or preco is None:
        return False

    categoria = categoria_combustivel(tipo)
    faixa = PRICE_SANITY_RANGES.get(categoria)

    if faixa is None:
        return False

    minimo, maximo = faixa
    return minimo <= preco <= maximo


def motivo_preco_descartado(tipo: str | None, preco: float | None) -> str | None:
    if tipo is None or preco is None:
        return None

    if preco_dentro_da_faixa(tipo, preco):
        return None

    categoria = categoria_combustivel(tipo)
    minimo, maximo = PRICE_SANITY_RANGES[categoria]
    return f"fora da faixa esperada ({minimo:.2f} a {maximo:.2f})"


def categoria_combustivel(tipo: str) -> str:
    if tipo == "etanol":
        return "etanol"
    if tipo in {"diesel", "diesel_s10"}:
        return "diesel"
    if tipo == "nao_identificado":
        return "nao_identificado"
    return "gasolina"


def nome_amigavel_combustivel(tipo: str | None) -> str:
    mapa = {
        "gasolina": "Gasolina",
        "gasolina_comum": "Gasolina comum",
        "gasolina_aditivada": "Gasolina aditivada",
        "gasolina_premium": "Gasolina premium",
        "etanol": "Etanol",
        "diesel": "Diesel",
        "diesel_s10": "Diesel S10",
        "nao_identificado": "Nao identificado",
    }
    return mapa.get(tipo or "", "Nao identificado")


def ordem_combustivel(tipo: str | None) -> int:
    ordem = {
        "gasolina_comum": 1,
        "gasolina_aditivada": 2,
        "gasolina_premium": 3,
        "etanol": 4,
        "diesel": 5,
        "diesel_s10": 6,
        "nao_identificado": 99,
    }
    return ordem.get(tipo or "", 99)


def formatar_preco(valor: float) -> str:
    return f"R$ {valor:.2f}"


def formatar_diferenca(valor: float) -> str:
    sinal = "+" if valor > 0 else "-"
    return f"{sinal}R$ {abs(valor):.2f}"


def formatar_diferenca_curta(valor: float) -> str:
    sinal = "+" if valor > 0 else "-"
    return f"{sinal}R${abs(valor):.2f}".replace(".", ",")


def formatar_localizacao(latitude, longitude) -> str | None:
    if latitude is None or longitude is None:
        return None

    try:
        return f"Lat {float(latitude):.5f}, Lon {float(longitude):.5f}"
    except (TypeError, ValueError):
        return None


def reverse_geocode(lat, lon) -> dict:
    if lat is None or lon is None:
        return {}

    try:
        latitude = float(lat)
        longitude = float(lon)
    except (TypeError, ValueError):
        return {}

    cache_key = (round(latitude, 5), round(longitude, 5))

    if cache_key in reverse_geocode_cache:
        return reverse_geocode_cache[cache_key]

    try:
        resposta = requests.get(
            NOMINATIM_REVERSE_URL,
            params={
                "lat": latitude,
                "lon": longitude,
                "format": "json",
            },
            headers={"User-Agent": "Posto360/1.0 reverse-geocoding"},
            timeout=4,
        )
        resposta.raise_for_status()
        dados = resposta.json()
        endereco = dados.get("address") or {}

        rua = endereco.get("road")
        bairro = endereco.get("suburb") or endereco.get("neighbourhood")
        cidade = endereco.get("city") or endereco.get("town")
        estado = endereco.get("state")
        principal = rua or bairro or cidade
        contexto = []

        if principal != bairro and bairro:
            contexto.append(bairro)
        if principal != cidade and cidade:
            contexto.append(cidade)

        endereco_formatado = principal or ""

        if contexto:
            endereco_formatado = f"{endereco_formatado}, {contexto[0]}" if endereco_formatado else contexto[0]
            if len(contexto) > 1:
                endereco_formatado = f"{endereco_formatado} - {contexto[1]}"

        resultado = {
            "endereco_formatado": endereco_formatado,
            "rua": rua,
            "bairro": bairro,
            "cidade": cidade,
            "estado": estado,
        }
        reverse_geocode_cache[cache_key] = resultado
        return resultado
    except Exception:
        reverse_geocode_cache[cache_key] = {}
        return {}


def resumir_localizacao(latitude, longitude, regiao: str | None = None) -> str | None:
    coordenadas = formatar_localizacao(latitude, longitude)

    if regiao and regiao != "Regiao nao informada":
        if coordenadas:
            return f"{regiao} (aprox.)"
        return regiao

    return coordenadas


def montar_localizacao_leitura(nome_arquivo: str) -> dict:
    localizacao = dados_dashboard["upload_localizacoes"].get(nome_arquivo) or {}
    latitude = localizacao.get("latitude")
    longitude = localizacao.get("longitude")
    endereco = reverse_geocode(latitude, longitude)
    regiao_manual = dados_dashboard["upload_regioes"].get(nome_arquivo, "Regiao nao informada")
    tem_endereco_automatico = any(
        endereco.get(campo) for campo in ("rua", "bairro", "cidade", "endereco_formatado")
    )
    fallback_manual = (
        regiao_manual
        if not tem_endereco_automatico and regiao_manual != "Regiao nao informada"
        else None
    )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "localizacao_detectada": formatar_localizacao(latitude, longitude),
        "localizacao_aproximada": fallback_manual,
        "endereco_formatado": endereco.get("endereco_formatado"),
        "rua": endereco.get("rua"),
        "bairro": endereco.get("bairro"),
        "cidade": endereco.get("cidade"),
        "estado": endereco.get("estado"),
    }


def texto_ia_valido(valor) -> str | None:
    if not isinstance(valor, str):
        return None

    texto = valor.strip()

    if not texto or normalizar_texto(texto) in {"null", "none", "nao identificado", "nao visivel"}:
        return None

    return texto


def combinar_localizacao_ia(localizacao: dict, leitura: dict) -> dict:
    endereco_provavel = texto_ia_valido(leitura.get("endereco_provavel"))
    bairro_provavel = texto_ia_valido(leitura.get("bairro_provavel"))
    cidade_provavel = texto_ia_valido(leitura.get("cidade_provavel"))
    referencia_visual = texto_ia_valido(leitura.get("referencia_visual"))

    localizacao["endereco_provavel"] = endereco_provavel
    localizacao["bairro_provavel"] = bairro_provavel
    localizacao["cidade_provavel"] = cidade_provavel
    localizacao["referencia_visual"] = referencia_visual

    if not localizacao.get("rua") and endereco_provavel:
        localizacao["rua"] = endereco_provavel

    if not localizacao.get("bairro") and bairro_provavel:
        localizacao["bairro"] = bairro_provavel

    if not localizacao.get("cidade") and cidade_provavel:
        localizacao["cidade"] = cidade_provavel

    if not localizacao.get("endereco_formatado") and endereco_provavel:
        partes = [endereco_provavel]
        bairro_cidade = " - ".join(
            parte for parte in (bairro_provavel, cidade_provavel) if parte
        )
        if bairro_cidade:
            partes.append(bairro_cidade)
        localizacao["endereco_formatado"] = ", ".join(partes)

    return localizacao


def posto_informado(posto: str | None) -> bool:
    return bool(posto and posto.strip() and posto != "Posto nao informado")


def normalizar_nome_posto(posto: str | None) -> str:
    return normalizar_texto(posto or "")


def distancia_aproximada(latitude_a, longitude_a, latitude_b, longitude_b) -> float | None:
    try:
        lat_a = float(latitude_a)
        lon_a = float(longitude_a)
        lat_b = float(latitude_b)
        lon_b = float(longitude_b)
        return ((lat_a - lat_b) ** 2 + (lon_a - lon_b) ** 2) ** 0.5
    except (TypeError, ValueError):
        return None


def sugerir_posto_para_leitura(item: dict) -> str:
    posto_atual = item.get("posto")
    if posto_informado(posto_atual):
        return str(posto_atual).strip()

    sugestoes_conhecidas = []

    for leitura in ordenar_por_recencia(dados_dashboard["leituras_imagem"]):
        if leitura.get("arquivo") == item.get("arquivo"):
            continue
        if posto_informado(leitura.get("posto")):
            sugestoes_conhecidas.append(leitura)

    for analise in ordenar_por_recencia(dados_dashboard["analises"]):
        if posto_informado(analise.get("posto")):
            sugestoes_conhecidas.append(
                {
                    "posto": analise.get("posto"),
                    "regiao": analise.get("regiao"),
                    "latitude": analise.get("latitude"),
                    "longitude": analise.get("longitude"),
                }
            )

    posto_provavel = (item.get("posto_provavel") or "").strip()
    nome_visivel = (item.get("nome_posto_visivel") or "").strip()
    regiao = item.get("regiao")
    latitude = item.get("latitude")
    longitude = item.get("longitude")

    if posto_provavel:
        alvo = normalizar_nome_posto(posto_provavel)
        for conhecido in sugestoes_conhecidas:
            if normalizar_nome_posto(conhecido.get("posto")) == alvo:
                return str(conhecido.get("posto")).strip()

    if nome_visivel:
        alvo = normalizar_nome_posto(nome_visivel)
        for conhecido in sugestoes_conhecidas:
            if normalizar_nome_posto(conhecido.get("posto")) == alvo:
                return str(conhecido.get("posto")).strip()

    if latitude is not None and longitude is not None:
        melhor_posto = None
        menor_distancia = None
        for conhecido in sugestoes_conhecidas:
            distancia = distancia_aproximada(
                latitude,
                longitude,
                conhecido.get("latitude"),
                conhecido.get("longitude"),
            )
            if distancia is None:
                continue
            if menor_distancia is None or distancia < menor_distancia:
                menor_distancia = distancia
                melhor_posto = conhecido.get("posto")

        if melhor_posto and menor_distancia is not None and menor_distancia <= 0.003:
            return str(melhor_posto).strip()

    if regiao and regiao != "Regiao nao informada":
        postos_mesma_regiao = []
        vistos = set()
        for conhecido in sugestoes_conhecidas:
            posto = conhecido.get("posto")
            if not posto_informado(posto):
                continue
            if conhecido.get("regiao") != regiao:
                continue
            chave = normalizar_nome_posto(posto)
            if chave in vistos:
                continue
            vistos.add(chave)
            postos_mesma_regiao.append(str(posto).strip())

        if len(postos_mesma_regiao) == 1:
            return postos_mesma_regiao[0]

    if posto_provavel:
        return posto_provavel
    if nome_visivel:
        return nome_visivel
    return ""


def enriquecer_leituras_com_posto(itens: List[dict]) -> None:
    for item in itens:
        item["posto_sugerido"] = sugerir_posto_para_leitura(item)
        item["precisa_confirmar_posto"] = not posto_informado(item.get("posto"))


def agora_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _racional_para_float(valor) -> float:
    if isinstance(valor, (int, float)):
        return float(valor)

    if hasattr(valor, "numerator") and hasattr(valor, "denominator") and valor.denominator:
        return float(valor.numerator) / float(valor.denominator)

    if isinstance(valor, tuple) and len(valor) == 2 and valor[1]:
        return float(valor[0]) / float(valor[1])

    return float(valor)


def _gps_para_decimal(coordenadas, referencia: str | None) -> float | None:
    try:
        graus = _racional_para_float(coordenadas[0])
        minutos = _racional_para_float(coordenadas[1])
        segundos = _racional_para_float(coordenadas[2])
        decimal = graus + (minutos / 60.0) + (segundos / 3600.0)

        if referencia in {"S", "W"}:
            decimal *= -1

        return round(decimal, 6)
    except Exception:
        return None


def extrair_contexto_exif(caminho_arquivo: str) -> dict:
    try:
        from PIL import ExifTags, Image
    except Exception:
        return {}

    try:
        with Image.open(caminho_arquivo) as imagem:
            exif = imagem.getexif()
    except Exception:
        return {}

    if not exif:
        return {}

    tags = {ExifTags.TAGS.get(chave, chave): valor for chave, valor in exif.items()}
    gps_bruto = tags.get("GPSInfo")
    data_foto = tags.get("DateTimeOriginal") or tags.get("DateTime")
    contexto = {}

    if data_foto:
        contexto["data_foto"] = str(data_foto)

    if gps_bruto:
        gps_tags = {ExifTags.GPSTAGS.get(chave, chave): valor for chave, valor in gps_bruto.items()}
        latitude = _gps_para_decimal(
            gps_tags.get("GPSLatitude"),
            gps_tags.get("GPSLatitudeRef"),
        )
        longitude = _gps_para_decimal(
            gps_tags.get("GPSLongitude"),
            gps_tags.get("GPSLongitudeRef"),
        )

        if latitude is not None and longitude is not None:
            contexto["gps_latitude"] = latitude
            contexto["gps_longitude"] = longitude

    return contexto


def openai_api_key_configurada() -> bool:
    return bool(OPENAI_API_KEY)


def erro_configuracao_openai() -> str | None:
    if openai_api_key_configurada():
        return None

    return OPENAI_CONFIG_ERROR_MESSAGE


def extrair_texto_resposta_openai(payload: dict) -> str:
    texto = payload.get("output_text")

    if isinstance(texto, str) and texto.strip():
        return texto

    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return content["text"]

    return ""


def limpar_json_markdown(texto: str) -> str:
    texto_limpo = (texto or "").strip()

    if texto_limpo.startswith("```"):
        texto_limpo = re.sub(r"^```(?:json)?\s*", "", texto_limpo)
        texto_limpo = re.sub(r"\s*```$", "", texto_limpo)

    return texto_limpo.strip()


def parse_resposta_openai_vision(texto: str):
    texto_limpo = limpar_json_markdown(texto)

    try:
        dados = json.loads(texto_limpo)
    except json.JSONDecodeError as erro:
        raise RuntimeError(f"resposta_openai_invalida={erro}")

    if isinstance(dados, list):
        return {"leituras": dados}

    if isinstance(dados, dict):
        return dados

    raise RuntimeError("resposta_openai_formato_nao_suportado")


def mapear_combustivel_ia(tipo: str | None) -> str | None:
    if not tipo:
        return None

    mapa = {
        "gasolina_comum": "gasolina_comum",
        "gasolina_aditivada": "gasolina_aditivada",
        "gasolina_premium": "gasolina_premium",
        "etanol": "etanol",
        "diesel": "diesel",
        "diesel_s10": "diesel_s10",
        "nao_identificado": "nao_identificado",
    }
    return mapa.get(tipo)


def nome_arquivo_processado(nome_arquivo: str) -> str:
    base, extensao = os.path.splitext(nome_arquivo)

    if extensao.lower() in {".heic", ".heif", ".webp", ".png"}:
        return f"{base}.jpg"

    return nome_arquivo


def formatos_upload_validos() -> set[str]:
    return {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".heic", ".heif"}


def escolher_leitura_principal(leituras: List[dict]) -> dict | None:
    leituras_validas = [
        item
        for item in leituras
        if item.get("tipo")
        and item.get("preco") is not None
        and isinstance(item.get("confianca"), (int, float))
    ]

    if not leituras_validas:
        return None

    return sorted(
        leituras_validas,
        key=lambda item: (item.get("confianca", 0), item.get("preco", 0)),
        reverse=True,
    )[0]


def normalizar_preco_ia(valor) -> float | None:
    if valor is None:
        return None

    if isinstance(valor, int):
        if 200 <= valor <= 1000:
            return round(float(valor) / 100.0, 3)
        return float(valor)

    if isinstance(valor, float):
        return round(valor, 3)

    if isinstance(valor, str):
        texto = valor.strip().replace(",", ".")
        if not texto:
            return None

        if texto.isdigit():
            numero = int(texto)
            if 200 <= numero <= 1000:
                return round(numero / 100.0, 3)

        try:
            return round(float(texto), 3)
        except ValueError:
            return None

    return None


def salvar_copia_debug_vision(caminho_arquivo: str) -> str | None:
    try:
        arquivos_existentes = [
            nome
            for nome in os.listdir(DEBUG_VISION_FOLDER)
            if nome.startswith("debug_") and nome.lower().endswith(".jpg")
        ]
        proximo_indice = len(arquivos_existentes) + 1
        nome_debug = f"debug_{proximo_indice:05d}.jpg"
        caminho_debug = os.path.join(DEBUG_VISION_FOLDER, nome_debug)
        shutil.copy2(caminho_arquivo, caminho_debug)
        print(f"[VISION] debug_imagem_salva={caminho_debug}")
        return caminho_debug
    except Exception as erro:
        print(f"[VISION] debug_imagem_erro={erro}")
        return None


def analisar_imagem_com_openai(caminho_arquivo: str) -> dict:
    nome_arquivo = os.path.basename(caminho_arquivo)
    contexto_exif = extrair_contexto_exif(caminho_arquivo)
    salvar_copia_debug_vision(caminho_arquivo)
    print(f"[VISION] iniciando leitura arquivo={nome_arquivo}")
    print(f"[VISION] arquivo={nome_arquivo} api_key_configurada={openai_api_key_configurada()}")
    print(f"[VISION] arquivo={nome_arquivo} modelo={OPENAI_VISION_MODEL}")
    print(f"[VISION] arquivo={nome_arquivo} contexto_exif={contexto_exif}")

    if not OPENAI_API_KEY:
        print(f"[CONFIG] ERRO: {OPENAI_CONFIG_ERROR_MESSAGE}")
        raise RuntimeError(OPENAI_CONFIG_ERROR_MESSAGE)

    extensao = os.path.splitext(caminho_arquivo.lower())[1]
    mime_type = "image/jpeg"

    if extensao == ".png":
        mime_type = "image/png"
    elif extensao == ".webp":
        mime_type = "image/webp"

    with open(caminho_arquivo, "rb") as arquivo_imagem:
        imagem_base64 = base64.b64encode(arquivo_imagem.read()).decode("utf-8")

    schema = {
        "type": "object",
        "properties": {
            "leituras": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "combustivel": {
                            "type": "string",
                            "enum": [
                                "gasolina_comum",
                                "gasolina_aditivada",
                                "gasolina_premium",
                                "etanol",
                                "diesel",
                                "diesel_s10",
                                "nao_identificado",
                            ],
                        },
                        "preco": {"anyOf": [{"type": "number"}, {"type": "string"}, {"type": "null"}]},
                        "confianca": {"anyOf": [{"type": "number"}, {"type": "string"}]},
                    },
                    "required": ["combustivel", "preco", "confianca"],
                    "additionalProperties": False,
                },
            },
            "endereco_provavel": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "bairro_provavel": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "cidade_provavel": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "referencia_visual": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": [
            "leituras",
            "endereco_provavel",
            "bairro_provavel",
            "cidade_provavel",
            "referencia_visual",
        ],
        "additionalProperties": False,
    }

    prompt = (
        "Pela imagem, extraia todos os combustiveis e precos visiveis. "
        "Tambem procure informacoes de localizacao visiveis na imagem: nome de rua ou avenida, bairro, cidade "
        "e alguma referencia visual do posto ou arredores quando estiver legivel. "
        "Retorne apenas JSON com a lista leituras e os campos endereco_provavel, bairro_provavel, cidade_provavel e referencia_visual. "
        "Formato: {\"leituras\":[{\"combustivel\":\"etanol\",\"preco\":4.99,\"confianca\":0.9}],\"endereco_provavel\":\"Av. do Contorno\",\"bairro_provavel\":\"Savassi\",\"cidade_provavel\":\"Belo Horizonte\",\"referencia_visual\":\"Posto Shell\"}. "
        "Inclua todos os precos visiveis. Nao retorne so um. Mesmo se tiver duvida, inclua. Nunca retorne vazio. "
        "Use combustivel entre: etanol, gasolina_comum, gasolina_aditivada, gasolina_premium, diesel, diesel_s10, nao_identificado. "
        "Se rua, bairro, cidade ou referencia nao estiverem visiveis, use null nesses campos. "
        "Aceite valores com virgula, ponto ou sem separador como 679 -> 6.79. "
        "Se vier markdown, o conteudo dentro dele deve continuar sendo um JSON valido."
    )

    body = {
        "model": OPENAI_VISION_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{imagem_base64}",
                        "detail": "high",
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "leitura_preco_posto360",
                "strict": True,
                "schema": schema,
            }
        },
    }

    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )

    print(f"[VISION] arquivo={nome_arquivo} iniciando_chamada_openai")

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            resposta = json.loads(response.read().decode("utf-8"))
            print(f"[VISION] arquivo={nome_arquivo} resposta_http_recebida=True")
            print(f"[VISION] resposta recebida arquivo={nome_arquivo}")
    except urllib.error.HTTPError as erro:
        detalhe = erro.read().decode("utf-8", errors="ignore")
        print(f"[VISION] erro arquivo={nome_arquivo} {erro}")
        print(f"[VISION] arquivo={nome_arquivo} erro_http={erro.code}")
        print(f"[VISION] arquivo={nome_arquivo} erro_http_detalhe={detalhe}")
        raise RuntimeError(f"falha_openai_http={erro.code} detalhe={detalhe[:240]}")
    except urllib.error.URLError as erro:
        print(f"[VISION] erro arquivo={nome_arquivo} {erro}")
        print(f"[VISION] arquivo={nome_arquivo} erro_rede={erro}")
        raise RuntimeError(f"falha_openai_rede={erro.reason}")
    except Exception as erro:
        print(f"[VISION] erro arquivo={nome_arquivo} {erro}")
        print(f"[VISION] arquivo={nome_arquivo} excecao_chamada={erro}")
        raise

    print(f"[VISION] arquivo={nome_arquivo} resposta_bruta_openai={json.dumps(resposta, ensure_ascii=False)}")
    texto_json = extrair_texto_resposta_openai(resposta)
    print(f"[VISION] arquivo={nome_arquivo} resposta_texto_bruta={texto_json}")

    if not texto_json:
        raise RuntimeError("resposta_openai_sem_texto")

    try:
        dados = parse_resposta_openai_vision(texto_json)
    except RuntimeError as erro:
        print(f"[VISION] arquivo={nome_arquivo} json_invalido={erro}")
        raise

    print(f"[VISION] arquivo={nome_arquivo} resposta_json={json.dumps(dados, ensure_ascii=False)}")

    return dados


def parse_data_registro(valor: str | None) -> datetime:
    if not valor:
        return datetime.now(timezone.utc)

    try:
        return datetime.fromisoformat(valor)
    except ValueError:
        return datetime.now(timezone.utc)


def manter_registros_recentes(registros: List[dict]) -> List[dict]:
    limite = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    return [item for item in registros if parse_data_registro(item.get("created_at")) >= limite]


def podar_historico_antigo() -> None:
    dados_dashboard["analises"] = manter_registros_recentes(dados_dashboard["analises"])
    dados_dashboard["leituras_imagem"] = manter_registros_recentes(dados_dashboard["leituras_imagem"])
    dados_dashboard["precos_imagem"] = [
        item
        for item in manter_registros_recentes(dados_dashboard["precos_imagem"])
        if item.get("preco") is not None
    ]

    arquivos_validos = {item["arquivo"] for item in dados_dashboard["leituras_imagem"]}
    dados_dashboard["upload_regioes"] = {
        arquivo: regiao
        for arquivo, regiao in dados_dashboard["upload_regioes"].items()
        if arquivo in arquivos_validos
    }
    dados_dashboard["upload_postos"] = {
        arquivo: posto
        for arquivo, posto in dados_dashboard["upload_postos"].items()
        if arquivo in arquivos_validos
    }
    dados_dashboard["upload_posto_ids"] = {
        arquivo: posto_id
        for arquivo, posto_id in dados_dashboard["upload_posto_ids"].items()
        if arquivo in arquivos_validos
    }
    dados_dashboard["upload_localizacoes"] = {
        arquivo: localizacao
        for arquivo, localizacao in dados_dashboard["upload_localizacoes"].items()
        if arquivo in arquivos_validos
    }
    dados_dashboard["upload_tipos"] = {
        arquivo: tipo
        for arquivo, tipo in dados_dashboard["upload_tipos"].items()
        if arquivo in arquivos_validos
    }


def listar_registros_precos() -> List[dict]:
    registros = []

    for item in dados_dashboard["analises"]:
        registros.append(
            {
                "origem": "formulario",
                "tipo": item["tipo"],
                "categoria": categoria_combustivel(item["tipo"]),
                "preco": item["preco"],
                "regiao": item["regiao"],
                "posto": item["posto"],
                "posto_id": item.get("posto_id"),
                "origem_preco": item.get("origem_preco", "meu_posto"),
                "created_at": item.get("created_at"),
            }
        )

    for item in dados_dashboard["precos_imagem"]:
        registros.append(
            {
                "origem": "imagem",
                "tipo": item["tipo"],
                "categoria": categoria_combustivel(item["tipo"]),
                "preco": item["preco"],
                "regiao": item["regiao"],
                "posto": item["posto"],
                "posto_id": item.get("posto_id"),
                "origem_preco": item.get("origem_preco", "concorrente"),
                "created_at": item.get("created_at"),
            }
        )

    return registros


def filtrar_por_regiao(itens: List[dict], regiao: str | None) -> List[dict]:
    if not regiao:
        return list(itens)

    return [item for item in itens if item.get("regiao") == regiao]


def filtrar_por_posto(itens: List[dict], posto: str | None) -> List[dict]:
    if not posto:
        return list(itens)

    return [item for item in itens if item.get("posto") == posto]


def ordenar_por_recencia(itens: List[dict]) -> List[dict]:
    return sorted(
        itens,
        key=lambda item: parse_data_registro(item.get("created_at")),
        reverse=True,
    )


def calcular_media_referencia(registro: dict, registros: List[dict]) -> float | None:
    mesma_regiao = [
        item["preco"]
        for item in registros
        if item["tipo"] == registro["tipo"]
        and item["regiao"] == registro["regiao"]
        and item["posto"] == registro["posto"]
    ]

    if mesma_regiao:
        return sum(mesma_regiao) / len(mesma_regiao)

    mesmo_posto = [
        item["preco"]
        for item in registros
        if item["tipo"] == registro["tipo"] and item["posto"] == registro["posto"]
    ]

    if mesmo_posto:
        return sum(mesmo_posto) / len(mesmo_posto)

    mesmo_tipo = [item["preco"] for item in registros if item["tipo"] == registro["tipo"]]

    if mesmo_tipo:
        return sum(mesmo_tipo) / len(mesmo_tipo)

    return None


def calcular_media_referencia_tipo(
    tipo: str,
    regiao: str | None,
    registros_base: List[dict],
) -> float | None:
    if regiao:
        precos_mesma_regiao = [
            item["preco"]
            for item in registros_base
            if item["tipo"] == tipo and item.get("regiao") == regiao
        ]
        if precos_mesma_regiao:
            return sum(precos_mesma_regiao) / len(precos_mesma_regiao)

    precos_tipo = [item["preco"] for item in registros_base if item["tipo"] == tipo]
    if precos_tipo:
        return sum(precos_tipo) / len(precos_tipo)

    return None


def classificar_preco(preco_atual: float, media_referencia: float) -> tuple[str, str, str]:
    diferenca = preco_atual - media_referencia

    if diferenca >= 0.25:
        return "muito acima", "baixar", formatar_diferenca(diferenca)
    if diferenca >= 0.08:
        return "acima da media", "revisar", formatar_diferenca(diferenca)
    if diferenca <= -0.25:
        return "muito abaixo", "oportunidade de aumentar margem", formatar_diferenca(diferenca)
    if diferenca <= -0.08:
        return "abaixo da media", "manter", formatar_diferenca(diferenca)
    return "na faixa", "manter", formatar_diferenca(diferenca)


def selecionar_registros_atuais(registros: List[dict]) -> List[dict]:
    atuais = {}

    for item in ordenar_por_recencia(registros):
        chave = (item["tipo"], item["regiao"], item["posto"])
        if chave not in atuais:
            atuais[chave] = item

    return list(atuais.values())


def gerar_comparativos_precos(registros_atuais: List[dict], registros: List[dict]) -> List[dict]:
    comparativos = []

    grupos = {}

    for item in registros_atuais:
        if item.get("tipo") not in COMBUSTIVEIS_DECISAO:
            continue

        chave = item["tipo"]
        grupos.setdefault(
            chave,
            {
                "tipo": item["tipo"],
                "combustivel_label": nome_amigavel_combustivel(item["tipo"]),
                "regiao": item.get("regiao"),
                "posto": item.get("posto"),
                "precos": [],
            },
        )
        grupos[chave]["precos"].append(item["preco"])

    for grupo in grupos.values():
        precos = grupo["precos"]
        media_grupo = sum(precos) / len(precos)
        media_referencia = calcular_media_referencia_tipo(
            grupo["tipo"],
            grupo.get("regiao"),
            registros,
        )

        if media_referencia is None:
            media_referencia = media_grupo

        status, recomendacao, diferenca_formatada = classificar_preco(
            media_grupo,
            media_referencia,
        )
        comparativos.append(
            {
                "combustivel": grupo["tipo"],
                "combustivel_label": grupo["combustivel_label"],
                "posto": grupo.get("posto"),
                "regiao": grupo.get("regiao"),
                "preco_atual": formatar_preco(media_grupo),
                "media_regiao": formatar_preco(media_referencia),
                "diferenca_valor": round(media_grupo - media_referencia, 2),
                "diferenca": diferenca_formatada,
                "status": status,
                "recomendacao": recomendacao,
                "quantidade_leituras": len(precos),
                "menor_preco": formatar_preco(min(precos)),
                "maior_preco": formatar_preco(max(precos)),
                "base": f"Baseado na media do grupo com {len(precos)} leitura(s)",
            }
        )

    return sorted(
        comparativos,
        key=lambda item: (
            ordem_combustivel(item.get("combustivel")),
            item.get("regiao") or "",
            item.get("posto") or "",
        ),
    )


def enriquecer_decisao_preco(item: dict) -> dict:
    diferenca = float(item.get("diferenca_valor") or 0)

    if abs(diferenca) < 0.005:
        diferenca_label = "Na média"
    else:
        diferenca_label = f"{formatar_diferenca_curta(diferenca)} vs média"

    if -0.03 <= diferenca <= 0.03:
        return {
            **item,
            "preco_sugerido": item.get("preco_atual"),
            "decisao_titulo": "Na média",
            "diferenca_media": diferenca_label,
            "decisao_classe": "decision-neutral",
        }

    if -0.15 <= diferenca <= -0.04:
        return {
            **item,
            "preco_sugerido": item.get("preco_atual"),
            "decisao_titulo": "Competitivo",
            "diferenca_media": diferenca_label,
            "decisao_classe": "decision-good",
        }

    if diferenca < -0.15:
        return {
            **item,
            "preco_sugerido": item.get("preco_atual"),
            "decisao_titulo": "Oportunidade de subir",
            "diferenca_media": diferenca_label,
            "decisao_classe": "decision-opportunity",
        }

    if 0.04 <= diferenca <= 0.15:
        return {
            **item,
            "preco_sugerido": item.get("preco_atual"),
            "decisao_titulo": "Levemente acima",
            "diferenca_media": diferenca_label,
            "decisao_classe": "decision-above-light",
        }

    if 0.16 <= diferenca <= 0.30:
        return {
            **item,
            "preco_sugerido": item.get("preco_atual"),
            "decisao_titulo": "Acima da média",
            "diferenca_media": diferenca_label,
            "decisao_classe": "decision-above",
        }

    return {
        **item,
        "preco_sugerido": item.get("preco_atual"),
        "decisao_titulo": "Risco de volume",
        "diferenca_media": diferenca_label,
        "decisao_classe": "decision-risk",
    }


def filtrar_comparativos_dashboard(comparativos: List[dict]) -> tuple[List[dict], bool]:
    visiveis = []
    ocultou_por_dados = False

    for item in comparativos:
        combustivel = item.get("combustivel")

        if combustivel not in COMBUSTIVEIS_DECISAO:
            ocultou_por_dados = True
            continue

        leituras_validas = int(item.get("quantidade_leituras") or 0)

        if combustivel not in COMBUSTIVEIS_CORE_DECISAO and leituras_validas < DECISION_MIN_VALID_READINGS:
            ocultou_por_dados = True
            continue

        visiveis.append(item)

    return visiveis, ocultou_por_dados


def horas_desde(valor: str | None) -> str:
    data = parse_data_registro(valor)
    agora = datetime.now(timezone.utc)

    if data.tzinfo is None:
        data = data.replace(tzinfo=timezone.utc)

    horas = max(0, int((agora - data).total_seconds() // 3600))

    if horas <= 0:
        return "Atualizado agora"
    if horas == 1:
        return "Atualizado há 1h"

    return f"Atualizado há {horas}h"


def montar_status_combustivel_posto(registro: dict, registros: List[dict]) -> dict:
    media_referencia = calcular_media_referencia_tipo(
        registro["tipo"],
        registro.get("regiao"),
        registros,
    )

    if media_referencia is None:
        media_referencia = registro["preco"]

    comparativo = enriquecer_decisao_preco(
        {
            "combustivel": registro["tipo"],
            "combustivel_label": nome_amigavel_combustivel(registro["tipo"]),
            "preco_atual": formatar_preco(registro["preco"]),
            "media_regiao": formatar_preco(media_referencia),
            "diferenca_valor": round(registro["preco"] - media_referencia, 2),
            "quantidade_leituras": 1,
        }
    )

    return {
        "label": nome_amigavel_combustivel(registro["tipo"]),
        "preco": formatar_preco(registro["preco"]),
        "status": comparativo["decisao_titulo"],
        "classe": comparativo["decisao_classe"],
    }


def selecionar_ultimo_registro_por_categoria(registros: List[dict], categoria: str) -> dict | None:
    registros_categoria = [
        item
        for item in registros
        if item.get("categoria") == categoria and item.get("tipo") in COMBUSTIVEIS_DECISAO
    ]

    if not registros_categoria:
        return None

    return ordenar_por_recencia(registros_categoria)[0]


def resumir_concorrentes_regiao(registros: List[dict], regiao: str, categoria: str) -> dict | None:
    precos = [
        item["preco"]
        for item in registros
        if item.get("origem_preco") == "concorrente"
        and item.get("regiao") == regiao
        and item.get("categoria") == categoria
    ]

    if not precos:
        return None

    return {
        "media": sum(precos) / len(precos),
        "menor": min(precos),
        "maior": max(precos),
        "leituras": len(precos),
    }


def classificar_vs_concorrentes(preco: float, media_concorrentes: float) -> tuple[str, str, str]:
    diferenca = round(preco - media_concorrentes, 2)

    if diferenca > 0.30:
        return "Risco de perder volume", "decision-risk", f"{formatar_diferenca_curta(diferenca)}"
    if diferenca > 0.08:
        return "Acima da região", "decision-above", f"{formatar_diferenca_curta(diferenca)}"
    if diferenca < -0.08:
        return "Competitivo", "decision-good", f"{formatar_diferenca_curta(diferenca)}"

    return "Na média", "decision-neutral", "Na média"


def montar_dados_postos() -> dict:
    registros = listar_registros_precos()
    postos = []

    for posto in dados_dashboard["postos_cadastrados"]:
        nome = posto.get("nome", "").strip()
        regiao_posto = posto.get("regiao") or ""
        registros_posto = [
            item
            for item in registros
            if normalizar_nome_posto(item.get("posto")) == normalizar_nome_posto(nome)
            and item.get("origem_preco") == "meu_posto"
        ]
        precos = []
        sem_concorrentes = False

        for categoria, label in (
            ("gasolina", "Gasolina"),
            ("etanol", "Etanol"),
            ("diesel", "Diesel"),
        ):
            registro_categoria = selecionar_ultimo_registro_por_categoria(registros_posto, categoria)
            if registro_categoria:
                concorrentes = resumir_concorrentes_regiao(registros, regiao_posto, categoria)
                if concorrentes:
                    status, classe, diferenca = classificar_vs_concorrentes(
                        registro_categoria["preco"],
                        concorrentes["media"],
                    )
                    precos.append(
                        {
                            "label": label,
                            "preco": formatar_preco(registro_categoria["preco"]),
                            "status": status,
                            "classe": classe,
                            "media_concorrentes": formatar_preco(concorrentes["media"]),
                            "menor_concorrente": formatar_preco(concorrentes["menor"]),
                            "maior_concorrente": formatar_preco(concorrentes["maior"]),
                            "diferenca": diferenca,
                        }
                    )
                else:
                    sem_concorrentes = True
                    precos.append(
                        {
                            "label": label,
                            "preco": formatar_preco(registro_categoria["preco"]),
                            "status": "Sem concorrentes",
                            "classe": "decision-neutral",
                            "media_concorrentes": "Sem dados",
                            "menor_concorrente": "Sem dados",
                            "maior_concorrente": "Sem dados",
                            "diferenca": "Sem dados",
                        }
                    )
            else:
                precos.append(
                    {
                        "label": label,
                        "preco": "Sem dados",
                        "status": "Sem leitura",
                        "classe": "decision-neutral",
                        "media_concorrentes": "Sem dados",
                        "menor_concorrente": "Sem dados",
                        "maior_concorrente": "Sem dados",
                        "diferenca": "Sem dados",
                    }
                )

        status_precos = [item["status"] for item in precos]
        if "Risco de perder volume" in status_precos:
            status_geral = "Crítico"
            status_classe = "station-critical"
        elif "Acima da região" in status_precos:
            status_geral = "Atenção"
            status_classe = "station-attention"
        else:
            status_geral = "OK"
            status_classe = "station-ok"

        alerta = next(
            (f"{item['label']} pode perder volume" for item in precos if item["status"] == "Risco de perder volume"),
            "",
        )
        if not alerta:
            alerta = next(
                (
                    f"{item['label']} está acima dos concorrentes da região."
                    for item in precos
                    if item["status"] == "Acima da região"
                ),
                "",
            )
        if not alerta and any(item["status"] in {"Competitivo", "Na média"} for item in precos):
            alerta = "Seu posto está competitivo hoje."

        ultima_leitura = max(
            (item.get("created_at") for item in registros_posto if item.get("created_at")),
            default=posto.get("created_at"),
        )

        postos.append(
            {
                "nome": nome,
                "endereco": posto.get("endereco") or "Endereço não informado",
                "regiao": regiao_posto,
                "status_geral": status_geral,
                "status_classe": status_classe,
                "atualizado": horas_desde(ultima_leitura),
                "precos": precos,
                "alerta": alerta,
                "sem_concorrentes": sem_concorrentes,
            }
        )

    return {"postos": postos}


def montar_dados_configuracoes() -> dict:
    configuracoes = dados_dashboard["configuracoes"]
    regioes = set(configuracoes.get("regioes", []))

    for regiao in listar_regioes_disponiveis():
        if regiao and regiao != "Regiao nao informada":
            regioes.add(regiao)

    return {
        "perfil": configuracoes["perfil"],
        "regioes": sorted(regioes),
        "parametros": configuracoes["parametros"],
        "alertas_config": configuracoes["alertas_config"],
        "plano": {
            "nome": "Básico",
            "postos_cadastrados": len(dados_dashboard["postos_cadastrados"]),
        },
    }


def buscar_posto_cadastrado(nome: str | None) -> dict | None:
    alvo = normalizar_nome_posto(nome)
    alvo_original = (nome or "").strip()

    if not alvo and not alvo_original:
        return None

    for posto in dados_dashboard["postos_cadastrados"]:
        if alvo_original and posto.get("id") == alvo_original:
            return posto
        if normalizar_nome_posto(posto.get("nome")) == alvo:
            return posto

    return None


def gerar_posto_id(nome: str, total_atual: int) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", normalizar_texto(nome)).strip("-")
    return f"{base or 'posto'}-{total_atual + 1}"


def posto_corresponde_registro(posto: dict, registro: dict) -> bool:
    posto_id = posto.get("id")

    if posto_id and registro.get("posto_id") == posto_id:
        return True

    return normalizar_nome_posto(registro.get("posto")) == normalizar_nome_posto(posto.get("nome"))


def formatar_diferenca_alerta(valor: float) -> str:
    diferenca = f"R${abs(valor):.2f}".replace(".", ",")
    sinal = "+" if valor > 0 else "-"

    if valor > 0:
        return f"+{diferenca}"
    if valor < 0:
        return f"-{diferenca}"

    return "R$0,00"


def montar_identificacao_alerta(item: dict) -> tuple[str, str]:
    posto = (item.get("posto") or "").strip()
    regiao = (item.get("regiao") or "").strip()

    if not posto or normalizar_texto(posto) == "posto nao informado":
        posto = "Posto não identificado"

    if not regiao or normalizar_texto(regiao) == "regiao nao informada":
        regiao = "Região não informada"

    return posto, regiao


def classificar_alerta_vs_concorrentes(diferenca: float) -> tuple[str, str, str, str, str, int]:
    if diferenca > 0.15:
        return (
            "Crítico",
            "alert-card-critical",
            "critical",
            "Pode estar perdendo volume",
            "Ação: avaliar redução de preço",
            1,
        )
    if 0.06 <= diferenca <= 0.15:
        return (
            "Atenção",
            "alert-card-attention",
            "attention",
            "Acima da média da região",
            "Ação: monitorar concorrentes",
            2,
        )
    if diferenca < -0.05:
        return (
            "Oportunidade",
            "alert-card-opportunity",
            "opportunity",
            "Abaixo da média da região",
            "Ação: pode subir preço sem perder competitividade",
            3,
        )

    return (
        "Manter preço",
        "alert-card-attention",
        "attention",
        "Na média",
        "Ação: manter preço",
        4,
    )


def gerar_alertas_precos(
    registros: List[dict],
    regiao_filtro: str | None = None,
    posto_filtro: str | None = None,
) -> List[dict]:
    alertas = []

    for posto in dados_dashboard["postos_cadastrados"]:
        nome_posto = (posto.get("nome") or "").strip()
        regiao_posto = (posto.get("regiao") or "").strip()

        if not nome_posto or not regiao_posto:
            continue
        if regiao_filtro and regiao_posto != regiao_filtro:
            continue
        if posto_filtro and nome_posto != posto_filtro:
            continue

        registros_posto = [
            item
            for item in registros
            if item.get("origem_preco") == "meu_posto"
            and posto_corresponde_registro(posto, item)
            and item.get("tipo") in COMBUSTIVEIS_DECISAO
        ]

        for categoria in ("gasolina", "etanol", "diesel"):
            registro_posto = selecionar_ultimo_registro_por_categoria(registros_posto, categoria)

            if not registro_posto:
                continue

            concorrentes = resumir_concorrentes_regiao(registros, regiao_posto, categoria)
            combustivel = nome_amigavel_combustivel(registro_posto.get("tipo"))

            if not concorrentes or concorrentes["leituras"] < ALERT_MIN_COMPETITOR_READINGS:
                alertas.append(
                    {
                        "titulo": "Sem dados",
                        "classe_dashboard": "alert-card-attention",
                        "classe_lista": "attention",
                        "posto": nome_posto,
                        "regiao": regiao_posto,
                        "combustivel_slug": registro_posto.get("tipo"),
                        "combustivel": combustivel,
                        "preco_meu_posto": formatar_preco(registro_posto["preco"]),
                        "media_concorrentes": "Sem dados",
                        "diferenca": "Sem dados",
                        "acao": "Sem dados suficientes de concorrentes na região",
                        "sugestao": "Ação: enviar leituras de concorrentes",
                        "mensagem": f"{nome_posto} · {combustivel} · Sem dados suficientes de concorrentes na região",
                        "prioridade": 5,
                    }
                )
                continue

            diferenca = round(registro_posto["preco"] - concorrentes["media"], 2)
            titulo, classe_dashboard, classe_lista, mensagem, sugestao, prioridade = classificar_alerta_vs_concorrentes(
                diferenca
            )
            diferenca_label = formatar_diferenca_alerta(diferenca)

            alertas.append(
                {
                    "titulo": titulo,
                    "classe_dashboard": classe_dashboard,
                    "classe_lista": classe_lista,
                    "posto": nome_posto,
                    "regiao": regiao_posto,
                    "combustivel_slug": registro_posto.get("tipo"),
                    "combustivel": combustivel,
                    "preco_meu_posto": formatar_preco(registro_posto["preco"]),
                    "media_concorrentes": formatar_preco(concorrentes["media"]),
                    "diferenca": diferenca_label,
                    "acao": mensagem,
                    "sugestao": sugestao,
                    "mensagem": f"{nome_posto} · {combustivel} · {diferenca_label} · {mensagem}",
                    "prioridade": prioridade,
                }
            )

    return sorted(
        alertas,
        key=lambda item: (
            item.get("prioridade", 99),
            item.get("posto") or "",
            ordem_combustivel(item.get("combustivel_slug")),
        ),
    )


def calcular_resumo_por_tipo(registros: List[dict]) -> dict:
    resumo = {
        "gasolina": {"label": "Gasolina", "media": "R$ --", "maior": "R$ --", "menor": "R$ --"},
        "etanol": {"label": "Etanol", "media": "R$ --", "maior": "R$ --", "menor": "R$ --"},
        "diesel": {"label": "Diesel", "media": "R$ --", "maior": "R$ --", "menor": "R$ --"},
    }

    for categoria in resumo:
        precos = [item["preco"] for item in registros if item["categoria"] == categoria]

        if precos:
            resumo[categoria] = {
                "label": resumo[categoria]["label"],
                "media": formatar_preco(sum(precos) / len(precos)),
                "maior": formatar_preco(max(precos)),
                "menor": formatar_preco(min(precos)),
            }

    return resumo


def listar_arquivos_uploads() -> List[str]:
    extensoes_validas = formatos_upload_validos()
    arquivos = []

    for nome_arquivo in os.listdir(UPLOAD_FOLDER):
        caminho = os.path.join(UPLOAD_FOLDER, nome_arquivo)
        extensao = os.path.splitext(nome_arquivo.lower())[1]

        if os.path.isfile(caminho) and extensao in extensoes_validas:
            arquivos.append(caminho)

    return sorted(arquivos)


def limpar_arquivos_uploads() -> int:
    total_removidos = 0

    for pasta_atual, _, arquivos in os.walk(UPLOAD_FOLDER):
        for nome_arquivo in arquivos:
            caminho = os.path.join(pasta_atual, nome_arquivo)
            try:
                os.remove(caminho)
                total_removidos += 1
            except OSError:
                print(f"[RESET] nao_foi_possivel_remover={caminho}")

    return total_removidos


def limpar_dados_teste() -> int:
    arquivos_removidos = limpar_arquivos_uploads()

    with PROCESSING_LOCK:
        dados_dashboard["reset_token"] += 1
        dados_dashboard["analises"] = []
        dados_dashboard["alertas"] = []
        dados_dashboard["precos_imagem"] = []
        dados_dashboard["leituras_imagem"] = []
        dados_dashboard["resumo"] = {}
        dados_dashboard["resumo_por_tipo"] = {}
        dados_dashboard["comparativos"] = []
        dados_dashboard["upload_regioes"] = {}
        dados_dashboard["upload_postos"] = {}
        dados_dashboard["upload_posto_ids"] = {}
        dados_dashboard["upload_localizacoes"] = {}
        dados_dashboard["upload_tipos"] = {}
        dados_dashboard["processamento_upload"] = {
            "em_andamento": False,
            "total": 0,
            "processadas": 0,
        }

    print(f"[RESET] dados_teste_limpos=True arquivos_removidos={arquivos_removidos}")
    return arquivos_removidos


def arquivo_eh_imagem(nome_arquivo: str) -> bool:
    return os.path.splitext(nome_arquivo.lower())[1] in formatos_upload_validos()


def salvar_imagem_optimizada(upload: UploadFile, caminho_destino: str) -> bool:
    nome_arquivo = upload.filename or os.path.basename(caminho_destino)
    formato_detectado = os.path.splitext(nome_arquivo.lower())[1].lstrip(".") or "desconhecido"

    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps

        upload.file.seek(0)
        upload.file.seek(0, 2)
        tamanho = upload.file.tell()
        upload.file.seek(0)

        print(f"[UPLOAD] arquivo={nome_arquivo} formato_detectado={formato_detectado}")
        print(f"[UPLOAD] arquivo={nome_arquivo} tamanho_original_bytes={tamanho}")

        if tamanho > MAX_IMAGE_BYTES:
            print(f"[UPLOAD] arquivo={nome_arquivo} rejeitado=tamanho_excedido")
            return False

        heic_suportado = False

        if formato_detectado in {"heic", "heif"}:
            try:
                import pillow_heif

                pillow_heif.register_heif_opener()
                heic_suportado = True
                print(f"[UPLOAD] arquivo={nome_arquivo} conversao_heic=ativa")
            except Exception:
                print(f"[UPLOAD] arquivo={nome_arquivo} conversao_heic=indisponivel")

        with Image.open(upload.file) as imagem:
            tamanho_original_px = imagem.size
            imagem_corrigida = ImageOps.exif_transpose(imagem)
            rotacao_corrigida = imagem_corrigida.size != imagem.size
            imagem = imagem_corrigida.convert("RGB")
            resampling = getattr(Image, "Resampling", Image)
            imagem.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), resampling.LANCZOS)
            imagem = ImageEnhance.Contrast(imagem).enhance(1.08)
            imagem = imagem.filter(ImageFilter.SHARPEN)

            print(f"[UPLOAD] arquivo={nome_arquivo} tamanho_original_px={tamanho_original_px}")
            print(f"[UPLOAD] arquivo={nome_arquivo} rotacao_corrigida={rotacao_corrigida}")
            print(f"[UPLOAD] arquivo={nome_arquivo} tamanho_apos_otimizacao_px={imagem.size}")

            imagem.save(
                caminho_destino,
                format="JPEG",
                optimize=True,
                quality=84,
            )

            tamanho_final = os.path.getsize(caminho_destino)
            print(f"[UPLOAD] arquivo={nome_arquivo} tamanho_final_bytes={tamanho_final}")
            print(f"[UPLOAD] arquivo={nome_arquivo} salvo_como=jpeg")

            if formato_detectado in {"heic", "heif"} and not heic_suportado:
                print(f"[UPLOAD] arquivo={nome_arquivo} observacao=heic_heif_aberto_sem_registro_explicito")

        return True
    except Exception:
        print(f"[UPLOAD] arquivo={nome_arquivo} erro_otimizacao=True")
        try:
            upload.file.seek(0)
            with open(caminho_destino, "wb") as buffer:
                shutil.copyfileobj(upload.file, buffer)
            print(f"[UPLOAD] arquivo={nome_arquivo} fallback_salvamento_bruto=True")
            return True
        except Exception:
            return False


def extrair_precos_do_texto(texto: str) -> List[float]:
    texto_normalizado = normalizar_texto(texto)
    texto_limpo = re.sub(r"[^\d,.\s]", " ", texto_normalizado)
    texto_limpo = re.sub(r"\s+", " ", texto_limpo).strip()
    candidatos = re.findall(r"(?<!\d)(\d{1,2}[.,]\d{2,3}|\d{3})(?!\d)", texto_limpo)
    precos = []

    for valor in candidatos:
        if valor.isdigit() and len(valor) == 3:
            preco = float(valor) / 100
        else:
            preco = float(valor.replace(",", "."))

        if 3 <= preco <= 9:
            precos.append(round(preco, 3))

    precos_unicos = []

    for preco in precos:
        if preco not in precos_unicos:
            precos_unicos.append(preco)

    return precos_unicos


def selecionar_preco_detectado(precos: List[float]) -> float | None:
    if not precos:
        return None

    precos_ordenados = []

    for preco in precos:
        if preco not in precos_ordenados:
            precos_ordenados.append(preco)

    return precos_ordenados[0]


def atualizar_resumo_dashboard() -> None:
    podar_historico_antigo()
    registros = listar_registros_precos()
    precos_base = [item["preco"] for item in registros]

    dados_dashboard["resumo"] = calcular_resumo(precos_base)
    dados_dashboard["resumo_por_tipo"] = calcular_resumo_por_tipo(registros)
    comparativos_brutos = [
        enriquecer_decisao_preco(item)
        for item in gerar_comparativos_precos(registros, registros)
    ]
    dados_dashboard["comparativos"], _ = filtrar_comparativos_dashboard(comparativos_brutos)
    dados_dashboard["alertas"] = calcular_alertas_operacionais(
        dados_dashboard["analises"]
    ) + gerar_alertas_precos(registros)


def listar_regioes_disponiveis() -> List[str]:
    regioes = set()

    for item in dados_dashboard["analises"]:
        if item.get("regiao"):
            regioes.add(item["regiao"])

    for item in dados_dashboard["leituras_imagem"]:
        if item.get("regiao") and item["regiao"] != "Regiao nao informada":
            regioes.add(item["regiao"])

    return sorted(regioes)


def listar_postos_disponiveis() -> List[str]:
    postos = set()

    for item in dados_dashboard["analises"]:
        if item.get("posto"):
            postos.add(item["posto"])

    for item in dados_dashboard["leituras_imagem"]:
        if item.get("posto") and item["posto"] != "Posto nao informado":
            postos.add(item["posto"])

    return sorted(postos)


def resumir_leituras_ia(leituras_imagem: List[dict]) -> dict:
    total_openai_vision = sum(1 for item in leituras_imagem if item.get("fonte") == "openai_vision")
    total_falhas = sum(1 for item in leituras_imagem if item.get("status") == "sem_leitura")
    total_lidas = len(leituras_imagem)
    mensagem_ia = None

    if total_openai_vision > 0:
        mensagem_ia = "Leitura por IA ativa"
    elif total_falhas > 0:
        mensagem_ia = "Falha na leitura. Envie imagem mais nitida"

    return {
        "total_openai_vision": total_openai_vision,
        "total_falhas": total_falhas,
        "total_lidas": total_lidas,
        "mensagem_ia": mensagem_ia,
    }


def montar_dados_dashboard_filtrados(regiao: str | None, posto: str | None) -> dict:
    podar_historico_antigo()
    analises = ordenar_por_recencia(
        filtrar_por_posto(filtrar_por_regiao(dados_dashboard["analises"], regiao), posto)
    )
    precos_imagem = ordenar_por_recencia(
        filtrar_por_posto(filtrar_por_regiao(dados_dashboard["precos_imagem"], regiao), posto)
    )
    leituras_imagem = ordenar_por_recencia(
        filtrar_por_posto(filtrar_por_regiao(dados_dashboard["leituras_imagem"], regiao), posto)
    )
    registros = []

    for item in analises:
        registros.append(
            {
                "origem": "formulario",
                "tipo": item["tipo"],
                "categoria": categoria_combustivel(item["tipo"]),
                "preco": item["preco"],
                "regiao": item["regiao"],
                "posto": item["posto"],
                "posto_id": item.get("posto_id"),
                "origem_preco": item.get("origem_preco", "meu_posto"),
                "created_at": item.get("created_at"),
            }
        )

    for item in precos_imagem:
        registros.append(
            {
                "origem": "imagem",
                "tipo": item["tipo"],
                "categoria": categoria_combustivel(item["tipo"]),
                "preco": item["preco"],
                "regiao": item["regiao"],
                "posto": item["posto"],
                "posto_id": item.get("posto_id"),
                "origem_preco": item.get("origem_preco", "concorrente"),
                "created_at": item.get("created_at"),
            }
        )

    comparativos_brutos = [
        enriquecer_decisao_preco(item)
        for item in gerar_comparativos_precos(registros, registros)
    ]
    comparativos, ocultou_combustiveis_por_dados = filtrar_comparativos_dashboard(comparativos_brutos)
    registros_alertas = listar_registros_precos()
    alertas = calcular_alertas_operacionais(analises) + gerar_alertas_precos(
        registros_alertas,
        regiao,
        posto,
    )
    resumo_por_tipo = calcular_resumo_por_tipo(registros)
    combustiveis_visiveis = [
        {"slug": slug, **dados}
        for slug, dados in resumo_por_tipo.items()
        if dados.get("media") and dados.get("media") != "R$ --"
    ]
    resumo_fontes_ia = resumir_leituras_ia(leituras_imagem)

    for item in analises:
        item["tipo_label"] = nome_amigavel_combustivel(item.get("tipo"))

    for item in leituras_imagem:
        item["tipo_label"] = nome_amigavel_combustivel(item.get("tipo"))
        for leitura in item.get("leituras_detectadas", []):
            leitura["tipo_label"] = nome_amigavel_combustivel(leitura.get("tipo"))
    enriquecer_leituras_com_posto(leituras_imagem)

    return {
        "analises": analises[:RECENT_ITEMS_LIMIT],
        "precos_imagem": precos_imagem,
        "leituras_imagem": leituras_imagem[:3],
        "alertas_principais": alertas[:3],
        "total_alertas": len(alertas),
        "mensagem_ia": resumo_fontes_ia["mensagem_ia"],
        "total_openai_vision": resumo_fontes_ia["total_openai_vision"],
        "total_lidas_imagem": resumo_fontes_ia["total_lidas"],
        "total_falhas_leitura": resumo_fontes_ia["total_falhas"],
        "resumo": calcular_resumo([item["preco"] for item in registros]),
        "resumo_por_tipo": resumo_por_tipo,
        "combustiveis_visiveis": combustiveis_visiveis,
        "comparativos": comparativos,
        "ocultou_combustiveis_por_dados": ocultou_combustiveis_por_dados,
        "alertas": alertas,
        "regioes": listar_regioes_disponiveis(),
        "postos": listar_postos_disponiveis(),
        "regiao_selecionada": regiao or "",
        "posto_selecionado": posto or "",
        "processamento_upload": dados_dashboard["processamento_upload"],
        "configuracao_alerta": erro_configuracao_openai(),
    }


def montar_dados_leituras(regiao: str | None, posto: str | None, page: int) -> dict:
    podar_historico_antigo()
    page = max(1, page)
    leituras_filtradas = ordenar_por_recencia(
        filtrar_por_posto(filtrar_por_regiao(dados_dashboard["leituras_imagem"], regiao), posto)
    )
    total = len(leituras_filtradas)
    inicio = (page - 1) * READINGS_PAGE_SIZE
    fim = inicio + READINGS_PAGE_SIZE
    itens = leituras_filtradas[inicio:fim]
    tem_mais = fim < total

    for item in itens:
        item["tipo_label"] = nome_amigavel_combustivel(item.get("tipo"))
        for leitura in item.get("leituras_detectadas", []):
            leitura["tipo_label"] = nome_amigavel_combustivel(leitura.get("tipo"))
    enriquecer_leituras_com_posto(itens)

    return {
        "leituras_imagem": itens,
        "total_lidas_imagem": total,
        "regioes": listar_regioes_disponiveis(),
        "postos": listar_postos_disponiveis(),
        "regiao_selecionada": regiao or "",
        "posto_selecionado": posto or "",
        "page": page,
        "tem_mais": tem_mais,
        "proxima_page": page + 1,
        "pagina_anterior": page - 1 if page > 1 else None,
    }


def montar_dados_alertas(regiao: str | None, posto: str | None) -> dict:
    dados = montar_dados_dashboard_filtrados(regiao, posto)
    return {
        "alertas": dados.get("alertas", []),
        "total_alertas": len(dados.get("alertas", [])),
        "regioes": dados.get("regioes", []),
        "postos": dados.get("postos", []),
        "regiao_selecionada": regiao or "",
        "posto_selecionado": posto or "",
    }


def precisa_reprocessar_uploads() -> bool:
    return not dados_dashboard["leituras_imagem"] and bool(listar_arquivos_uploads())


def processar_arquivo_visao_ia(caminho_arquivo: str) -> dict:
    nome_arquivo = os.path.basename(caminho_arquivo)
    localizacao_leitura = montar_localizacao_leitura(nome_arquivo)
    origem_preco = dados_dashboard["upload_tipos"].get(nome_arquivo, "concorrente")
    print(f"[VISION] iniciando leitura arquivo={nome_arquivo}")
    try:
        leitura = analisar_imagem_com_openai(caminho_arquivo)
        status_leitura = leitura.get("status", "ok")
        confianca = leitura.get("confianca")
        confianca_contexto = leitura.get("confianca_contexto")
        bandeira = leitura.get("bandeira")
        nome_posto_visivel = leitura.get("nome_posto_visivel")
        posto_provavel = leitura.get("posto_provavel")
        regiao_provavel = leitura.get("regiao_provavel")
        endereco_provavel = texto_ia_valido(leitura.get("endereco_provavel"))
        bairro_provavel = texto_ia_valido(leitura.get("bairro_provavel"))
        cidade_provavel = texto_ia_valido(leitura.get("cidade_provavel"))
        referencia_visual = texto_ia_valido(leitura.get("referencia_visual"))
        localizacao_leitura = combinar_localizacao_ia(localizacao_leitura, leitura)
        leituras_detectadas = []
        leituras_descartadas = []
        leituras_brutas = leitura.get("leituras", [])

        if not isinstance(leituras_brutas, list):
            leituras_brutas = []

        print(f"[VISION] arquivo={nome_arquivo} leituras_brutas_antes_validacao={json.dumps(leituras_brutas, ensure_ascii=False)}")

        for item in leituras_brutas:
            combustivel_item_bruto = item.get("combustivel")
            tipo_item = mapear_combustivel_ia(combustivel_item_bruto)
            preco_item = normalizar_preco_ia(item.get("preco"))
            confianca_item = item.get("confianca")

            motivo_descarte = None

            if not isinstance(confianca_item, (int, float)):
                try:
                    confianca_item = float(str(confianca_item).replace(",", "."))
                except (TypeError, ValueError):
                    confianca_item = 0.0

            if not tipo_item:
                tipo_item = "nao_identificado"

            if not isinstance(preco_item, float):
                motivo_descarte = "preco_invalido"
            elif not 2.0 <= preco_item <= 10.0:
                motivo_descarte = "fora_da_faixa_debug_2_10"

            if motivo_descarte:
                leituras_descartadas.append(
                    {
                        "combustivel": combustivel_item_bruto,
                        "tipo": tipo_item,
                        "preco": preco_item,
                        "confianca": round(float(confianca_item), 2),
                        "motivo": motivo_descarte,
                    }
                )
                print(
                    f"[VISION] arquivo={nome_arquivo} leitura_descartada combustivel={combustivel_item_bruto} "
                    f"tipo={tipo_item} preco={preco_item} confianca={round(float(confianca_item), 2)} motivo={motivo_descarte}"
                )
            else:
                leituras_detectadas.append(
                    {
                        "combustivel": combustivel_item_bruto,
                        "tipo": tipo_item,
                        "preco": round(preco_item, 3),
                        "confianca": round(float(confianca_item), 2),
                    }
                )
                print(
                    f"[VISION] arquivo={nome_arquivo} leitura_aceita combustivel={combustivel_item_bruto} "
                    f"tipo={tipo_item} preco={round(preco_item, 3)} confianca={round(float(confianca_item), 2)}"
                )

        leitura_principal = escolher_leitura_principal(leituras_detectadas)
        tipo_combustivel = leitura_principal["tipo"] if leitura_principal else "nao_identificado"
        preco = leitura_principal["preco"] if leitura_principal else None

        print(f"[VISION] arquivo={nome_arquivo} status={status_leitura}")
        print(f"[VISION] arquivo={nome_arquivo} confianca_bruta={confianca}")
        print(f"[VISION] arquivo={nome_arquivo} leituras_detectadas={leituras_detectadas}")
        print(f"[VISION] arquivo={nome_arquivo} leituras_descartadas={leituras_descartadas}")
        print(f"[VISION] arquivo={nome_arquivo} bandeira={bandeira}")
        print(f"[VISION] arquivo={nome_arquivo} posto_provavel={posto_provavel}")
        print(f"[VISION] arquivo={nome_arquivo} regiao_provavel={regiao_provavel}")
        print(f"[VISION] arquivo={nome_arquivo} endereco_provavel={endereco_provavel}")
        print(f"[VISION] arquivo={nome_arquivo} bairro_provavel={bairro_provavel}")
        print(f"[VISION] arquivo={nome_arquivo} cidade_provavel={cidade_provavel}")
        print(f"[VISION] arquivo={nome_arquivo} referencia_visual={referencia_visual}")
        print(f"[VISION] arquivo={nome_arquivo} confianca_contexto={confianca_contexto}")

        if not isinstance(confianca, (int, float)):
            confianca = 0.0

        if isinstance(preco, int):
            preco = float(preco)

        if not isinstance(confianca_contexto, (int, float)):
            confianca_contexto = None

        if leitura_principal:
            confianca = max(float(confianca or 0), float(leitura_principal["confianca"]))

        if not isinstance(preco, float):
            raise RuntimeError("Nao foi possivel identificar o preco")

        if not isinstance(confianca, (int, float)):
            confianca = 0.0

        if tipo_combustivel is None:
            tipo_combustivel = "nao_identificado"

        print(f"[VISION] arquivo={nome_arquivo} tipo_mapeado={tipo_combustivel}")
        print(f"[VISION] arquivo={nome_arquivo} preco_principal={preco}")
        print(f"[VISION] arquivo={nome_arquivo} confianca_final={confianca:.2f}")

        confirmacao_necessaria = False
        mensagem_contexto = ""
        contexto_detectado = any(
            [
                bandeira,
                nome_posto_visivel,
                posto_provavel,
                regiao_provavel,
                endereco_provavel,
                bairro_provavel,
                cidade_provavel,
                referencia_visual,
            ]
        )
        status_final = "lido_com_sucesso"

        if OPENAI_VISION_MIN_CONFIDENCE <= float(confianca) < OPENAI_VISION_LOW_CONFIDENCE:
            status_final = "baixa_confianca"
            mensagem_contexto = "Leitura com baixa confianca. Confira a imagem no detalhe."

        return {
            "arquivo": nome_arquivo,
            "tipo": tipo_combustivel,
            "preco": round(preco, 3),
            "confianca": round(float(confianca), 2),
            "fonte": "openai_vision",
            "mensagem": mensagem_contexto,
            "leituras_detectadas": leituras_detectadas,
            "leituras_descartadas": leituras_descartadas,
            "bandeira": bandeira,
            "nome_posto_visivel": nome_posto_visivel,
            "posto_provavel": posto_provavel,
            "regiao_provavel": regiao_provavel,
            "endereco_provavel": endereco_provavel,
            "bairro_provavel": bairro_provavel,
            "cidade_provavel": cidade_provavel,
            "referencia_visual": referencia_visual,
            "confianca_contexto": round(float(confianca_contexto), 2) if isinstance(confianca_contexto, (int, float)) else None,
            "confirmacao_necessaria": confirmacao_necessaria,
            "regiao": dados_dashboard["upload_regioes"].get(
                nome_arquivo, "Regiao nao informada"
            ),
            "posto": dados_dashboard["upload_postos"].get(
                nome_arquivo, "Posto nao informado"
            ),
            "posto_id": dados_dashboard["upload_posto_ids"].get(nome_arquivo),
            "origem_preco": origem_preco,
            **localizacao_leitura,
            "status": status_final,
            "created_at": agora_iso(),
        }
    except Exception as erro_visao:
        print(f"[VISION] erro arquivo={nome_arquivo} {erro_visao}")
        return {
            "arquivo": nome_arquivo,
            "tipo": "nao_identificado",
            "preco": None,
            "confianca": None,
            "fonte": "openai_vision",
            "mensagem": "Foto sem leitura confiavel. Tire mais perto e com menos reflexo.",
            "leituras_detectadas": [],
            "leituras_descartadas": [],
            "bandeira": None,
            "nome_posto_visivel": None,
            "posto_provavel": None,
            "regiao_provavel": None,
            "endereco_provavel": None,
            "bairro_provavel": None,
            "cidade_provavel": None,
            "referencia_visual": None,
            "confianca_contexto": None,
            "confirmacao_necessaria": False,
            "regiao": dados_dashboard["upload_regioes"].get(
                nome_arquivo, "Regiao nao informada"
            ),
            "posto": dados_dashboard["upload_postos"].get(
                nome_arquivo, "Posto nao informado"
            ),
            "posto_id": dados_dashboard["upload_posto_ids"].get(nome_arquivo),
            "origem_preco": origem_preco,
            **localizacao_leitura,
            "status": "sem_leitura",
            "created_at": agora_iso(),
        }


def atualizar_resultados_imagem(resultado: dict) -> None:
    dados_dashboard["leituras_imagem"] = [
        item for item in dados_dashboard["leituras_imagem"] if item["arquivo"] != resultado["arquivo"]
    ]
    dados_dashboard["leituras_imagem"].append(resultado)

    precos_normalizados = []

    for item in dados_dashboard["leituras_imagem"]:
        if item["status"] not in {"lido_com_sucesso", "baixa_confianca"}:
            continue

        leituras_detectadas = item.get("leituras_detectadas") or []

        if leituras_detectadas:
            for leitura in leituras_detectadas:
                precos_normalizados.append(
                    {
                        "arquivo": item["arquivo"],
                        "tipo": leitura["tipo"],
                        "preco": leitura["preco"],
                        "confianca": leitura.get("confianca"),
                        "fonte": item.get("fonte"),
                        "regiao": item["regiao"],
                        "posto": item["posto"],
                        "posto_id": item.get("posto_id"),
                        "origem_preco": item.get("origem_preco", "concorrente"),
                        "latitude": item.get("latitude"),
                        "longitude": item.get("longitude"),
                        "endereco_formatado": item.get("endereco_formatado"),
                        "rua": item.get("rua"),
                        "bairro": item.get("bairro"),
                        "cidade": item.get("cidade"),
                        "estado": item.get("estado"),
                        "endereco_provavel": item.get("endereco_provavel"),
                        "bairro_provavel": item.get("bairro_provavel"),
                        "cidade_provavel": item.get("cidade_provavel"),
                        "referencia_visual": item.get("referencia_visual"),
                        "created_at": item["created_at"],
                    }
                )
        elif item["preco"] is not None:
            precos_normalizados.append(
                {
                    "arquivo": item["arquivo"],
                    "tipo": item["tipo"],
                    "preco": item["preco"],
                    "confianca": item.get("confianca"),
                    "fonte": item.get("fonte"),
                    "regiao": item["regiao"],
                    "posto": item["posto"],
                    "posto_id": item.get("posto_id"),
                    "origem_preco": item.get("origem_preco", "concorrente"),
                    "latitude": item.get("latitude"),
                    "longitude": item.get("longitude"),
                    "endereco_formatado": item.get("endereco_formatado"),
                    "rua": item.get("rua"),
                    "bairro": item.get("bairro"),
                    "cidade": item.get("cidade"),
                    "estado": item.get("estado"),
                    "endereco_provavel": item.get("endereco_provavel"),
                    "bairro_provavel": item.get("bairro_provavel"),
                    "cidade_provavel": item.get("cidade_provavel"),
                    "referencia_visual": item.get("referencia_visual"),
                    "created_at": item["created_at"],
                }
            )

    dados_dashboard["precos_imagem"] = precos_normalizados


def processar_uploads_com_ia(caminhos_arquivo: List[str] | None = None) -> None:
    caminhos = caminhos_arquivo or listar_arquivos_uploads()
    token_processamento = dados_dashboard.get("reset_token", 0)
    if not openai_api_key_configurada():
        print(f"[CONFIG] ERRO: {OPENAI_CONFIG_ERROR_MESSAGE}")
        with PROCESSING_LOCK:
            dados_dashboard["processamento_upload"] = {
                "em_andamento": False,
                "total": len(caminhos),
                "processadas": 0,
            }
        return

    with PROCESSING_LOCK:
        dados_dashboard["processamento_upload"]["em_andamento"] = True
        dados_dashboard["processamento_upload"]["total"] = len(caminhos)
        dados_dashboard["processamento_upload"]["processadas"] = 0

    max_workers = max(1, min(VISION_MAX_WORKERS, len(caminhos)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {
            executor.submit(processar_arquivo_visao_ia, caminho_arquivo): caminho_arquivo
            for caminho_arquivo in caminhos
        }

        for futuro in as_completed(futuros):
            try:
                resultado = futuro.result()
                with PROCESSING_LOCK:
                    if token_processamento == dados_dashboard.get("reset_token", 0):
                        atualizar_resultados_imagem(resultado)
            except Exception:
                pass
            finally:
                with PROCESSING_LOCK:
                    if token_processamento == dados_dashboard.get("reset_token", 0):
                        dados_dashboard["processamento_upload"]["processadas"] += 1

    with PROCESSING_LOCK:
        if token_processamento == dados_dashboard.get("reset_token", 0):
            dados_dashboard["processamento_upload"]["em_andamento"] = False
            atualizar_resumo_dashboard()


def iniciar_processamento_em_thread(caminhos: List[str]) -> None:
    if not caminhos:
        with PROCESSING_LOCK:
            dados_dashboard["processamento_upload"] = {
                "em_andamento": False,
                "total": 0,
                "processadas": 0,
        }
        return

    if not openai_api_key_configurada():
        print(f"[CONFIG] ERRO: {OPENAI_CONFIG_ERROR_MESSAGE}")
        with PROCESSING_LOCK:
            dados_dashboard["processamento_upload"] = {
                "em_andamento": False,
                "total": len(caminhos),
                "processadas": 0,
            }
        return

    thread = threading.Thread(
        target=processar_uploads_com_ia,
        args=(caminhos,),
        daemon=True,
    )
    thread.start()


@app.get("/")
def home():
    return {"mensagem": "Posto360 rodando 🚀"}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    if (
        openai_api_key_configurada()
        and precisa_reprocessar_uploads()
        and not dados_dashboard["processamento_upload"]["em_andamento"]
    ):
        iniciar_processamento_em_thread(listar_arquivos_uploads())
    elif not openai_api_key_configurada():
        print(f"[CONFIG] ERRO: {OPENAI_CONFIG_ERROR_MESSAGE}")

    regiao = request.query_params.get("regiao") or None
    posto = request.query_params.get("posto") or None
    dados_filtrados = montar_dados_dashboard_filtrados(regiao, posto)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"dados": dados_filtrados},
    )


@app.post("/reset")
def reset_dados_teste():
    limpar_dados_teste()
    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/leituras", response_class=HTMLResponse)
def leituras(request: Request):
    regiao = request.query_params.get("regiao") or None
    posto = request.query_params.get("posto") or None
    try:
        page = int(request.query_params.get("page", "1"))
    except ValueError:
        page = 1

    dados_leituras = montar_dados_leituras(regiao, posto, page)

    return templates.TemplateResponse(
        request=request,
        name="leituras.html",
        context={"dados": dados_leituras},
    )


@app.get("/alertas", response_class=HTMLResponse)
def alertas(request: Request):
    regiao = request.query_params.get("regiao") or None
    posto = request.query_params.get("posto") or None
    dados_alertas = montar_dados_alertas(regiao, posto)

    return templates.TemplateResponse(
        request=request,
        name="alertas.html",
        context={"dados": dados_alertas},
    )


@app.get("/postos", response_class=HTMLResponse)
def postos(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="postos.html",
        context={"dados": montar_dados_postos()},
    )


@app.post("/postos")
def salvar_posto(
    nome: str = Form(""),
    endereco: str = Form(""),
    regiao: str = Form(""),
):
    nome_limpo = nome.strip()

    if nome_limpo:
        dados_dashboard["postos_cadastrados"].append(
            {
                "id": gerar_posto_id(nome_limpo, len(dados_dashboard["postos_cadastrados"])),
                "nome": nome_limpo,
                "endereco": endereco.strip(),
                "regiao": regiao.strip(),
                "created_at": agora_iso(),
            }
        )

    return RedirectResponse(url="/postos", status_code=303)


@app.get("/configuracoes", response_class=HTMLResponse)
def configuracoes(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="configuracoes.html",
        context={"dados": montar_dados_configuracoes()},
    )


@app.post("/configuracoes")
def salvar_configuracoes(
    nome: str = Form(""),
    email: str = Form(""),
    empresa: str = Form(""),
    nova_regiao: str = Form(""),
    margem_acima_media: str = Form("0.05"),
    margem_alerta_critico: str = Form("0.10"),
    estrategia: str = Form("Equilibrado"),
    preco_acima_media: str | None = Form(None),
    concorrentes_reduziram: str | None = Form(None),
    oportunidade_margem: str | None = Form(None),
):
    configuracoes = dados_dashboard["configuracoes"]
    configuracoes["perfil"] = {
        "nome": nome.strip(),
        "email": email.strip(),
        "empresa": empresa.strip(),
    }

    regiao_limpa = nova_regiao.strip()
    if regiao_limpa and regiao_limpa not in configuracoes["regioes"]:
        configuracoes["regioes"].append(regiao_limpa)

    configuracoes["parametros"] = {
        "margem_acima_media": margem_acima_media.strip() or "0.05",
        "margem_alerta_critico": margem_alerta_critico.strip() or "0.10",
        "estrategia": estrategia if estrategia in {"Conservador", "Equilibrado", "Agressivo"} else "Equilibrado",
    }
    configuracoes["alertas_config"] = {
        "preco_acima_media": preco_acima_media == "on",
        "concorrentes_reduziram": concorrentes_reduziram == "on",
        "oportunidade_margem": oportunidade_margem == "on",
    }

    return RedirectResponse(url="/configuracoes", status_code=303)


@app.get("/upload-processando", response_class=HTMLResponse)
def upload_processando(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="upload_processando.html",
        context={"dados": dados_dashboard},
    )


@app.post("/confirmar-posto")
def confirmar_posto(
    arquivo: str = Form(""),
    posto_nome: str = Form(""),
    redirect_to: str = Form("/dashboard"),
):
    posto_limpo = posto_nome.strip()

    if arquivo and posto_limpo:
        dados_dashboard["upload_postos"][arquivo] = posto_limpo

        for item in dados_dashboard["leituras_imagem"]:
            if item.get("arquivo") == arquivo:
                item["posto"] = posto_limpo
                item["confirmacao_necessaria"] = False
                break

        for item in dados_dashboard["precos_imagem"]:
            if item.get("arquivo") == arquivo:
                item["posto"] = posto_limpo

        atualizar_resumo_dashboard()

    destino = redirect_to or "/dashboard"
    return RedirectResponse(url=destino, status_code=303)


@app.get("/formulario", response_class=HTMLResponse)
def formulario(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="formulario.html",
        context={},
    )


@app.post("/formulario")
def enviar_formulario(
    posto: str = Form(""),
    regiao: str = Form(""),
    produto: str = Form(""),
    preco: float = Form(0),
    litros_vendidos: float = Form(0),
    litros_estoque: float = Form(0),
):
    dias_restantes = 0

    if litros_vendidos > 0:
        dias_restantes = litros_estoque / litros_vendidos

    tipo_combustivel = identificar_tipo_combustivel(produto)
    posto_cadastrado = buscar_posto_cadastrado(posto)

    dados_dashboard["analises"].append(
        {
            "posto": posto,
            "regiao": regiao,
            "produto": produto,
            "tipo": tipo_combustivel,
            "preco": round(preco, 2),
            "posto_id": posto_cadastrado.get("id") if posto_cadastrado else None,
            "origem_preco": "meu_posto",
            "dias_restantes": round(dias_restantes, 2),
            "created_at": agora_iso(),
        }
    )

    atualizar_resumo_dashboard()

    return RedirectResponse(url="/dashboard", status_code=303)


@app.post("/upload-imagens")
async def upload_imagens(
    regiao: str = Form(""),
    tipo_leitura: str = Form("concorrente"),
    posto_cliente: str = Form(""),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    imagens: List[UploadFile] = File(...),
):
    caminhos = []
    tipo_leitura = "meu_posto" if tipo_leitura == "meu_posto" else "concorrente"
    posto_cadastrado = buscar_posto_cadastrado(posto_cliente)
    regiao_upload = regiao.strip()

    if tipo_leitura == "meu_posto" and posto_cadastrado:
        regiao_upload = regiao_upload or posto_cadastrado.get("regiao", "")

    for imagem in imagens[:MAX_UPLOAD_IMAGES]:
        if not imagem.filename or not arquivo_eh_imagem(imagem.filename):
            continue

        nome_salvo = nome_arquivo_processado(imagem.filename)
        caminho = f"{UPLOAD_FOLDER}/{nome_salvo}"
        dados_dashboard["upload_tipos"][nome_salvo] = tipo_leitura
        dados_dashboard["upload_regioes"][nome_salvo] = regiao_upload or "Regiao nao informada"
        if tipo_leitura == "meu_posto" and posto_cadastrado:
            dados_dashboard["upload_postos"][nome_salvo] = posto_cadastrado.get("nome", "Posto nao informado")
            dados_dashboard["upload_posto_ids"][nome_salvo] = posto_cadastrado.get("id")
        dados_dashboard["upload_localizacoes"][nome_salvo] = {
            "latitude": latitude,
            "longitude": longitude,
        }

        if not salvar_imagem_optimizada(imagem, caminho):
            continue

        caminhos.append(caminho)

    with PROCESSING_LOCK:
        dados_dashboard["processamento_upload"] = {
            "em_andamento": bool(caminhos),
            "total": len(caminhos),
            "processadas": 0,
        }
    iniciar_processamento_em_thread(caminhos)

    return RedirectResponse(url="/upload-processando", status_code=303)


@app.get("/upload-teste", response_class=HTMLResponse)
def upload_teste(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="upload_teste.html",
        context={"postos": dados_dashboard["postos_cadastrados"]},
    )
