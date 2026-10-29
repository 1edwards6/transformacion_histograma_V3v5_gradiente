# -*- coding: utf-8 -*-
"""Núcleo matemático de V3v5: histograma, regiones locales y cuadrícula por gradiente.

Basado en el cuaderno fuente entregado por el usuario. La etapa de búsqueda de la
cuadrícula se ha optimizado para no construir DataFrames en cada posición candidata.
"""
import math
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import PchipInterpolator

niveles = np.arange(256)

def calcular_magnitud_gradiente(imagen_gris, suavizado=0.8):
    """Calcula |gradiente I| = sqrt((dI/dx)^2 + (dI/dy)^2).

    Se usa una imagen en escala de grises y un suavizado gaussiano opcional para
    reducir la influencia del ruido fino antes de calcular las derivadas.
    """
    img = np.asarray(imagen_gris, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("El cálculo del gradiente requiere una imagen en escala de grises.")
    if min(img.shape) < 2:
        raise ValueError("La imagen debe tener al menos 2 filas y 2 columnas.")
    if float(suavizado) > 0:
        img = cv2.GaussianBlur(
            img, (0, 0), sigmaX=float(suavizado), sigmaY=float(suavizado)
        )
    gradiente_y, gradiente_x = np.gradient(img)
    return np.hypot(gradiente_x, gradiente_y).astype(np.float32)


def _imagen_integral(valores):
    """Imagen integral: permite obtener sumas rectangulares en tiempo constante."""
    valores = np.asarray(valores, dtype=np.float64)
    return np.pad(valores, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)


def _media_rectangulo(integral, x1, y1, x2, y2):
    """Media de una región rectangular [x1:x2, y1:y2] usando una imagen integral."""
    total = (
        integral[y2, x2] - integral[y1, x2]
        - integral[y2, x1] + integral[y1, x1]
    )
    area = max(1, (x2 - x1) * (y2 - y1))
    return float(total / area)


def _pares_candidatos_3x3(dimension, paso=5.0, banda_minima=10.0,
                          region_minima=30.0):
    """Pares de cortes para la cuadrícula exploratoria 3×3.

    `banda_minima` limita las bandas exteriores; `region_minima` limita el ancho
    de la región central. Si los parámetros se contradicen, se reduce la banda
    efectiva para asegurar que exista al menos una pareja válida.
    """
    if dimension < 3:
        return []
    paso = max(float(paso), 1.0)
    region_minima = float(np.clip(region_minima, 1.0, 80.0))
    banda_minima = float(np.clip(banda_minima, 1.0, 40.0))
    banda_minima = min(banda_minima, max(1.0, (100.0 - region_minima) / 2.0))

    limite_superior = 100.0 - banda_minima
    porcentajes = np.arange(banda_minima, limite_superior + 1e-9, paso)
    if len(porcentajes) == 0 or porcentajes[-1] < limite_superior - 1e-9:
        porcentajes = np.append(porcentajes, limite_superior)
    porcentajes = np.unique(np.clip(porcentajes, banda_minima, limite_superior))

    pares = []
    vistos = set()
    for p1 in porcentajes:
        for p2 in porcentajes:
            if p2 - p1 + 1e-9 < region_minima:
                continue
            x1 = int(np.clip(round(dimension * p1 / 100.0), 1, dimension - 2))
            x2 = int(np.clip(round(dimension * p2 / 100.0), x1 + 1, dimension - 1))
            if x2 <= x1:
                continue
            clave = (x1, x2)
            if clave not in vistos:
                pares.append((float(p1), float(p2), x1, x2))
                vistos.add(clave)
    return pares


def _medias_celdas_3x3(integral, ancho, alto, x1, x2, y1, y2):
    """Devuelve gradiente medio de cada celda de la cuadrícula exploratoria 3×3."""
    xs = [0, int(x1), int(x2), int(ancho)]
    ys = [0, int(y1), int(y2), int(alto)]
    medias = []
    registros = []
    for fila in range(3):
        fila_medias = []
        for columna in range(3):
            media = _media_rectangulo(
                integral, xs[columna], ys[fila], xs[columna + 1], ys[fila + 1]
            )
            fila_medias.append(media)
            registros.append({
                "Fila": fila + 1,
                "Columna": columna + 1,
                "Celda": f"({fila + 1}, {columna + 1})",
                "Gradiente medio": media,
                "Tipo": "Central candidata" if (fila, columna) == (1, 1) else "Exterior",
            })
        medias.append(fila_medias)
    return medias, pd.DataFrame(registros)


def seleccionar_cuadricula_adaptativa(imagen_gris, paso_busqueda=5.0,
                                      banda_minima=10.0, region_minima=30.0,
                                      peso_exterior=0.55, suavizado=2.0):
    """Busca la región candidata central 3×3 con gradiente alto relativo al exterior.

    Se maximiza J = G_central - peso_exterior * media(G_exteriores). Los dos
    ejes definitivos 2×2 se colocan en el punto medio geométrico de esa región.
    La búsqueda usa una imagen integral; solo se crea la tabla diagnóstica para
    la mejor candidata, lo que evita un gran número de DataFrames temporales.
    """
    img = np.asarray(imagen_gris, dtype=np.uint8)
    if img.ndim != 2:
        raise ValueError("La selección de cuadrícula requiere una imagen en escala de grises.")
    alto, ancho = img.shape
    if alto < 3 or ancho < 3:
        raise ValueError("La imagen debe tener al menos 3 filas y 3 columnas.")

    gradiente = calcular_magnitud_gradiente(img, suavizado=suavizado)
    integral = _imagen_integral(gradiente)
    region_minima_efectiva = float(np.clip(region_minima, 1.0, 80.0))
    banda_minima_efectiva = min(
        float(np.clip(banda_minima, 1.0, 40.0)),
        max(1.0, (100.0 - region_minima_efectiva) / 2.0)
    )
    pares_x = _pares_candidatos_3x3(ancho, paso_busqueda, banda_minima_efectiva, region_minima_efectiva)
    pares_y = _pares_candidatos_3x3(alto, paso_busqueda, banda_minima_efectiva, region_minima_efectiva)

    if not pares_x:
        x1, x2 = max(1, ancho // 3), min(ancho - 1, max(2, 2 * ancho // 3))
        pares_x = [(100.0*x1/ancho, 100.0*x2/ancho, x1, x2)]
    if not pares_y:
        y1, y2 = max(1, alto // 3), min(alto - 1, max(2, 2 * alto // 3))
        pares_y = [(100.0*y1/alto, 100.0*y2/alto, y1, y2)]

    mejor_puntaje = -float("inf")
    mejor = None
    mejor_distancia_centro = float("inf")
    peso = float(peso_exterior)
    for x1_pct, x2_pct, x1, x2 in pares_x:
        xs = [0, x1, x2, ancho]
        for y1_pct, y2_pct, y1, y2 in pares_y:
            ys = [0, y1, y2, alto]
            # Las nueve medias se calculan directamente desde la imagen integral.
            g = [[_media_rectangulo(integral, xs[c], ys[r], xs[c+1], ys[r+1])
                  for c in range(3)] for r in range(3)]
            central = float(g[1][1])
            exterior = float(np.mean([g[r][c] for r in range(3) for c in range(3)
                                      if (r, c) != (1, 1)]))
            puntaje = central - peso * exterior
            distancia_centro = abs(((x1 + x2) / 2.0) / ancho - 0.5) + abs(((y1 + y2) / 2.0) / alto - 0.5)
            tolerancia = 1e-10 * max(1.0, abs(puntaje), abs(mejor_puntaje) if np.isfinite(mejor_puntaje) else 1.0)
            mejora_real = mejor is None or puntaje > mejor_puntaje + tolerancia
            empate_mas_centrado = (mejor is not None and abs(puntaje - mejor_puntaje) <= tolerancia
                                   and distancia_centro < mejor_distancia_centro)
            # En imágenes planas o cuando las candidatas empatan, preferir una zona centrada
            # evita colocar la cuadrícula arbitrariamente cerca de un borde.
            if mejora_real or empate_mas_centrado:
                mejor_puntaje = puntaje
                mejor_distancia_centro = distancia_centro
                mejor = {
                    "x1_pct": x1_pct, "x2_pct": x2_pct,
                    "y1_pct": y1_pct, "y2_pct": y2_pct,
                    "x1_px": x1, "x2_px": x2, "y1_px": y1, "y2_px": y2,
                    "gradiente_central": central,
                    "gradiente_exterior": exterior,
                    "puntaje": float(puntaje),
                }

    x1, x2 = int(mejor["x1_px"]), int(mejor["x2_px"])
    y1, y2 = int(mejor["y1_px"]), int(mejor["y2_px"])
    corte_x = int(np.clip(round((x1 + x2) / 2.0), 1, ancho - 1))
    corte_y = int(np.clip(round((y1 + y2) / 2.0), 1, alto - 1))
    _, tabla = _medias_celdas_3x3(integral, ancho, alto, x1, x2, y1, y2)

    return {
        "corte_vertical_px": corte_x,
        "corte_horizontal_px": corte_y,
        "corte_vertical_pct": 100.0 * corte_x / ancho,
        "corte_horizontal_pct": 100.0 * corte_y / alto,
        "region_candidata_3x3": {
            "x1_pct": mejor["x1_pct"], "x2_pct": mejor["x2_pct"],
            "y1_pct": mejor["y1_pct"], "y2_pct": mejor["y2_pct"],
            "x1_px": x1, "x2_px": x2, "y1_px": y1, "y2_px": y2,
        },
        "gradiente_medio_central": mejor["gradiente_central"],
        "gradiente_medio_exterior": mejor["gradiente_exterior"],
        "puntaje": mejor["puntaje"],
        "tabla_gradientes_3x3": tabla,
        "magnitud_gradiente": gradiente,
        "banda_minima_efectiva_pct": banda_minima_efectiva,
        "region_minima_pct": region_minima_efectiva,
    }

def calcular_metricas(img):
    img = np.asarray(img, dtype=np.uint8)
    hist = np.bincount(img.ravel(), minlength=256)
    probabilidades = hist / img.size
    probabilidades = probabilidades[probabilidades > 0]
    entropia = -np.sum(probabilidades * np.log2(probabilidades))
    p1, p99 = np.percentile(img, [1, 99])
    return {
        "Media": float(np.mean(img)),
        "Desviación estándar": float(np.std(img)),
        "Mínimo": int(img.min()),
        "Máximo": int(img.max()),
        "Rango dinámico": int(img.max() - img.min()),
        "Rango P1-P99": float(p99 - p1),
        "Entropía (bits)": float(entropia),
        "Píxeles en 0 (%)": float(100 * np.mean(img == 0)),
        "Píxeles en 255 (%)": float(100 * np.mean(img == 255)),
    }


def encontrar_intervalo_minimo(hist, porcentaje):
    """Intervalo más corto que cubre el porcentaje indicado.

    Desempates: menor ancho, después mayor cantidad de píxeles y luego menor inicio.
    Devuelve r_min, r_max, píxeles dentro, total y píxeles requeridos.
    """
    h = np.asarray(hist, dtype=np.int64).ravel()
    if h.size != 256:
        raise ValueError("El histograma debe tener 256 frecuencias.")
    if np.any(h < 0):
        raise ValueError("Las frecuencias no pueden ser negativas.")
    if not (0 < float(porcentaje) <= 100):
        raise ValueError("El porcentaje debe ser mayor que 0 y como máximo 100.")
    total = int(h.sum())
    if total <= 0:
        raise ValueError("El histograma está vacío.")

    requeridos = int(math.ceil(total * float(porcentaje) / 100.0))
    prefijos = np.concatenate(([0], np.cumsum(h, dtype=np.int64)))
    candidatos = []
    for inicio in range(256):
        objetivo_prefijo = int(prefijos[inicio]) + requeridos
        if objetivo_prefijo > total:
            continue
        fin_exclusivo = int(np.searchsorted(prefijos, objetivo_prefijo, side="left"))
        final = fin_exclusivo - 1
        if final < inicio or final > 255:
            continue
        cantidad = int(prefijos[final + 1] - prefijos[inicio])
        ancho = final - inicio + 1
        candidatos.append((ancho, -cantidad, inicio, final, cantidad))
    if not candidatos:
        raise RuntimeError("No se encontró un intervalo válido.")
    ancho, cantidad_negativa, r_min, r_max, dentro = min(candidatos)
    return int(r_min), int(r_max), int(dentro), total, int(requeridos)


def aproximar_histograma(histograma_rango, metodo="PCHIP", num_puntos=250, num_armonicos=12):
    """Aproxima el histograma seleccionado siguiendo las opciones de la V3."""
    h = np.asarray(histograma_rango, dtype=float).ravel()
    n = len(h)
    coordenadas = np.arange(n, dtype=float)
    metodo = metodo.upper()
    if n == 1:
        return h.copy()

    if metodo in ("PCHIP", "LINEAL"):
        puntos_efectivos = min(max(2, int(num_puntos)), n)
        puntos_x = np.unique(np.rint(np.linspace(0, n - 1, puntos_efectivos)).astype(int))
        puntos_y = h[puntos_x]
        if metodo == "PCHIP":
            interpolante = PchipInterpolator(puntos_x, puntos_y, extrapolate=False)
            aproximado = np.asarray(interpolante(coordenadas), dtype=float)
        else:
            aproximado = np.interp(coordenadas, puntos_x, puntos_y)
    elif metodo == "FOURIER":
        if not 0 <= int(num_armonicos) <= n // 2:
            # Para regiones de tamaño variable limitamos el número de armónicos efectivo.
            num_armonicos = min(max(0, int(num_armonicos)), n // 2)
        coeficientes = np.fft.rfft(h)
        max_armonicos = len(coeficientes) - 1
        k = min(max(0, int(num_armonicos)), max_armonicos)
        coeficientes_filtrados = np.zeros_like(coeficientes)
        coeficientes_filtrados[:k + 1] = coeficientes[:k + 1]
        aproximado = np.fft.irfft(coeficientes_filtrados, n=n)
    else:
        raise ValueError("METODO_APROXIMACION debe ser PCHIP, LINEAL o FOURIER.")
    return np.maximum(np.asarray(aproximado, dtype=float), 0.0)


def frecuencias_realmente_asignadas(frecuencias_objetivo, total_pixeles):
    """Reproduce la regla V3: recorre los niveles de salida y asigna los sobrantes a 255."""
    frecuencias = np.maximum(np.asarray(frecuencias_objetivo, dtype=np.int64).ravel(), 0)
    if frecuencias.size != 256:
        raise ValueError("Se requieren 256 frecuencias objetivo.")
    logradas = np.zeros(256, dtype=np.int64)
    cursor = 0
    total = int(total_pixeles)
    for nivel in range(256):
        disponibles = total - cursor
        if disponibles <= 0:
            break
        cantidad = min(int(frecuencias[nivel]), disponibles)
        if cantidad > 0:
            logradas[nivel] += cantidad
            cursor += cantidad
    if cursor < total:
        logradas[255] += total - cursor
    return logradas


def calcular_objetivo_v3(hist, porcentaje=99.5, metodo="PCHIP", num_puntos=250, num_armonicos=12):
    """Aplica el cálculo de frecuencias objetivo de la V3 a un histograma dado.

    A, B, C y D se calculan desde ese histograma y su número de píxeles.
    Devuelve histograma objetivo, distribución realmente asignada y diagnóstico.
    """
    h = np.asarray(hist, dtype=np.int64).ravel()
    if h.size != 256 or np.any(h < 0) or h.sum() <= 0:
        raise ValueError("Se requiere un histograma válido de 256 niveles y suma positiva.")
    N = int(h.sum())
    r_min, r_max, pixeles_dentro, total, requeridos = encontrar_intervalo_minimo(h, porcentaje)
    h_rango = h[r_min:r_max + 1].astype(float).copy()
    longitud = len(h_rango)
    coordenadas_locales = np.arange(longitud, dtype=float)
    h_aprox = aproximar_histograma(h_rango, metodo, num_puntos, num_armonicos)

    rd = int(r_max - r_min)
    A = 255.0 / rd if rd > 0 else 1.0
    B = -float(r_min)
    D = 0.0

    niveles_float = np.arange(256, dtype=float)
    coordenada_fuente = (niveles_float - r_min - B) / A
    soporte = (coordenada_fuente >= 0) & (coordenada_fuente <= longitud - 1)
    valores_fuente = np.interp(coordenada_fuente, coordenadas_locales, h_aprox, left=0.0, right=0.0)
    base = np.zeros(256, dtype=float)
    base[soporte] = np.maximum(valores_fuente[soporte], 0.0)
    suma_base = float(base.sum())

    if suma_base <= 0:
        # Salvaguarda si una aproximación patológica produce área nula.
        objetivo = h.copy()
        logradas = frecuencias_realmente_asignadas(objetivo, N)
        diagnostico = {
            "r_min": r_min, "r_max": r_max, "pixeles_dentro": pixeles_dentro,
            "porcentaje_real": 100.0 * pixeles_dentro / N,
            "A": A, "B": B, "C": 1.0, "D": D, "N": N,
            "fallback": "histograma original por área transformada nula",
            "histograma_aproximado": h_aprox,
            "cdf_entrada": np.cumsum(h) / N,
            "cdf_salida": np.cumsum(logradas) / N,
        }
        return objetivo, logradas, diagnostico

    def calcular_frecuencias(c):
        return np.rint(c * base).astype(np.int64)

    def suma_frecuencias(c):
        return int(calcular_frecuencias(c).sum())

    C_area_total = N / suma_base
    inferior = 0.0
    superior = max(C_area_total, np.finfo(float).tiny)
    while suma_frecuencias(superior) < N:
        superior *= 2.0
    for _ in range(200):
        medio = inferior + (superior - inferior) / 2.0
        if medio == inferior or medio == superior:
            break
        if suma_frecuencias(medio) >= N:
            superior = medio
        else:
            inferior = medio
    C = superior
    for _ in range(1000):
        anterior = np.nextafter(C, -np.inf)
        if anterior <= 0 or suma_frecuencias(anterior) < N:
            break
        C = anterior

    funcion_transformada = np.maximum(C * base + D, 0.0)
    objetivo = np.maximum(np.rint(funcion_transformada).astype(np.int64), 0)
    logradas = frecuencias_realmente_asignadas(objetivo, N)

    diagnostico = {
        "r_min": r_min, "r_max": r_max, "pixeles_dentro": pixeles_dentro,
        "porcentaje_real": 100.0 * pixeles_dentro / N,
        "A": A, "B": B, "C": C, "D": D, "N": N,
        "C_area_total": C_area_total, "suma_base": suma_base,
        "suma_objetivo": int(objetivo.sum()),
        "suma_asignada": int(logradas.sum()), "fallback": "",
        "histograma_aproximado": h_aprox,
        "cdf_entrada": np.cumsum(h) / N,
        "cdf_salida": np.cumsum(logradas) / N,
    }
    return objetivo, logradas, diagnostico


def reasignar_pixeles_por_frecuencia(imagen_gris, frecuencias_objetivo):
    """Reasignación ordenada exacta usada por la V3 global."""
    original = np.asarray(imagen_gris, dtype=np.uint8)
    plano = original.ravel()
    total = plano.size
    logradas = frecuencias_realmente_asignadas(frecuencias_objetivo, total)
    orden = np.argsort(plano, kind="stable")
    salida = np.empty_like(plano)
    cursor = 0
    for nivel, cantidad in enumerate(logradas):
        cantidad = int(cantidad)
        if cantidad:
            salida[orden[cursor:cursor + cantidad]] = nivel
            cursor += cantidad
    return salida.reshape(original.shape), np.bincount(salida, minlength=256)

def crear_mapa_intensidad(hist_entrada, hist_salida_logrado, proteger_constante=True):
    """Crea T(r): una sola intensidad de salida por cada intensidad de entrada.

    Regla de desambiguación: entre todos los niveles de salida a los que la V3
    asignaría píxeles de una misma intensidad de entrada, se elige la mediana inferior:
    índice (n - 1) // 2. Para intensidades de entrada no observadas se interpola T.
    """
    h_in = np.asarray(hist_entrada, dtype=np.int64).ravel()
    h_out = np.asarray(hist_salida_logrado, dtype=np.int64).ravel()
    if h_in.size != 256 or h_out.size != 256:
        raise ValueError("Ambos histogramas deben tener 256 niveles.")
    N = int(h_in.sum())
    if N <= 0 or int(h_out.sum()) != N:
        raise ValueError("La suma de los histogramas debe ser el mismo total positivo.")

    niveles_entrada_presentes = np.flatnonzero(h_in > 0)
    if proteger_constante and len(niveles_entrada_presentes) == 1:
        # No hay variación local que realzar; conservar la región constante.
        return np.arange(256, dtype=float), {
            "niveles_entrada_presentes": niveles_entrada_presentes,
            "salidas_posibles_por_intensidad": {int(niveles_entrada_presentes[0]): [int(niveles_entrada_presentes[0])]},
            "representantes": {int(niveles_entrada_presentes[0]): int(niveles_entrada_presentes[0])},
            "region_constante_protegida": True,
            "cantidad_ambiguas": 0,
        }

    # Rango acumulado [inicio, fin) que ocupa cada nivel de salida en la lista ordenada.
    intervalos_salida = []
    cursor = 0
    for nivel_salida, cantidad in enumerate(h_out):
        cantidad = int(cantidad)
        if cantidad > 0:
            intervalos_salida.append((cursor, cursor + cantidad, int(nivel_salida)))
            cursor += cantidad
    if cursor != N:
        raise RuntimeError("Los intervalos de salida no cubren todos los píxeles.")

    cdf_entrada_entera = np.cumsum(h_in, dtype=np.int64)
    salidas_posibles = {}
    representantes = {}
    inicio_entrada = 0
    for nivel_entrada in niveles_entrada_presentes:
        fin_entrada = int(cdf_entrada_entera[nivel_entrada])
        posibles = [
            nivel_salida
            for inicio_salida, fin_salida, nivel_salida in intervalos_salida
            if max(inicio_entrada, inicio_salida) < min(fin_entrada, fin_salida)
        ]
        if not posibles:
            raise RuntimeError(f"No se hallaron salidas para la intensidad {nivel_entrada}.")
        # Los niveles son distintos y ya vienen ordenados de menor a mayor.
        indice_mediana_inferior = (len(posibles) - 1) // 2
        representante = int(posibles[indice_mediana_inferior])
        salidas_posibles[int(nivel_entrada)] = [int(v) for v in posibles]
        representantes[int(nivel_entrada)] = representante
        inicio_entrada = fin_entrada

    x = np.asarray(list(representantes.keys()), dtype=float)
    y = np.asarray(list(representantes.values()), dtype=float)
    mapa = np.interp(np.arange(256, dtype=float), x, y, left=y[0], right=y[-1])
    ambiguas = sum(len(v) > 1 for v in salidas_posibles.values())
    return mapa, {
        "niveles_entrada_presentes": niveles_entrada_presentes,
        "salidas_posibles_por_intensidad": salidas_posibles,
        "representantes": representantes,
        "region_constante_protegida": False,
        "cantidad_ambiguas": int(ambiguas),
    }

def _indices_y_pesos_interpolacion(coordenadas, centros):
    coordenadas = np.asarray(coordenadas, dtype=float)
    centros = np.asarray(centros, dtype=float)
    if len(centros) == 1:
        ceros = np.zeros(coordenadas.shape, dtype=int)
        return ceros, ceros, np.zeros(coordenadas.shape, dtype=float)
    idx_alto = np.searchsorted(centros, coordenadas, side="right")
    idx_alto = np.clip(idx_alto, 0, len(centros) - 1)
    idx_bajo = np.maximum(idx_alto - 1, 0)
    denom = centros[idx_alto] - centros[idx_bajo]
    peso_alto = np.zeros(coordenadas.shape, dtype=float)
    validos = denom > 0
    peso_alto[validos] = (coordenadas[validos] - centros[idx_bajo[validos]]) / denom[validos]
    peso_alto = np.clip(peso_alto, 0.0, 1.0)
    peso_alto[coordenadas <= centros[0]] = 0.0
    peso_alto[coordenadas >= centros[-1]] = 1.0
    idx_bajo[coordenadas <= centros[0]] = 0
    idx_alto[coordenadas <= centros[0]] = 0
    idx_bajo[coordenadas >= centros[-1]] = len(centros) - 1
    idx_alto[coordenadas >= centros[-1]] = len(centros) - 1
    peso_alto[coordenadas >= centros[-1]] = 0.0
    return idx_bajo, idx_alto, peso_alto


def convertir_a_rgb_con_luminancia(imagen_rgb_original, imagen_intensidad, es_color):
    """Devuelve gris o RGB; si hay color, sustituye el canal L del espacio Lab."""
    intensidad = np.asarray(imagen_intensidad, dtype=np.uint8)
    if not es_color:
        return intensidad
    imagen_lab = cv2.cvtColor(np.asarray(imagen_rgb_original, dtype=np.uint8), cv2.COLOR_RGB2LAB)
    imagen_lab[:, :, 0] = intensidad
    return cv2.cvtColor(imagen_lab, cv2.COLOR_LAB2RGB)


def procesar_imagen_v3v5(imagen_rgb_original, parametros=None):
    """Ejecuta V3 global + V3 local + fusión adaptativa + selección 2×2 por gradiente.

    La V3 central calcula el histograma objetivo al expandir/reacomodar la
    distribución y reasigna las intensidades manteniendo el total de píxeles.
    V3v2 aplica mapas locales por región e interpola espacialmente las tablas.
    La etapa más reciente busca por gradiente la región candidata 3×3 y coloca
    los dos ejes finales en su punto medio para trabajar con cuatro cuadrantes.
    """
    p = dict(parametros or {})
    rgb = np.asarray(imagen_rgb_original, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("La imagen debe tener tres canales RGB.")
    if min(rgb.shape[:2]) < 3:
        raise ValueError("La imagen debe tener como mínimo 3 × 3 píxeles.")

    es_color = bool(np.any(np.max(rgb, axis=2) != np.min(rgb, axis=2)))
    imagen = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)
    alto, ancho = imagen.shape

    porcentaje = float(p.get("porcentaje_conservado", 97.5))
    metodo = str(p.get("metodo", "PCHIP")).upper()
    puntos = int(p.get("num_puntos", 150))
    armonicos = int(p.get("num_armonicos", 12))
    paso = float(p.get("paso_busqueda", 5.0))
    banda = float(p.get("banda_minima", 10.0))
    region_min = float(p.get("region_minima", 30.0))
    peso_ext = float(p.get("peso_exterior", 0.55))
    suavizado = float(p.get("suavizado_gradiente", 2.0))
    fusionar = bool(p.get("fusionar_celdas", True))
    umbral = float(p.get("umbral_desviacion", 10.0))
    proteger_constantes = bool(p.get("proteger_constantes", True))
    clahe_limite = float(p.get("clahe_limite", 2.0))
    clahe_grid = int(p.get("clahe_grid", 8))

    # V3 GLOBAL: algoritmo original de objetivo de histograma + reasignación por frecuencia.
    # Salvaguarda: si la imagen completa es constante, no hay contraste que redistribuir;
    # se conserva la identidad en vez de producir una salida artificialmente negra.
    histograma_global = np.bincount(imagen.ravel(), minlength=256).astype(np.int64)
    if np.count_nonzero(histograma_global) <= 1:
        nivel_constante = int(np.flatnonzero(histograma_global)[0])
        objetivo_global = histograma_global.copy()
        logrado_global = histograma_global.copy()
        imagen_v3_global = imagen.copy()
        histograma_v3_global = histograma_global.copy()
        total_global = int(histograma_global.sum())
        cdf_constante = np.cumsum(histograma_global) / total_global
        diagnostico_global = {
            "r_min": nivel_constante, "r_max": nivel_constante,
            "pixeles_dentro": total_global, "porcentaje_real": 100.0,
            "A": 1.0, "B": -float(nivel_constante), "C": 1.0, "D": 0.0,
            "N": total_global, "fallback": "imagen constante: se conserva la transformación identidad",
            "cdf_entrada": cdf_constante, "cdf_salida": cdf_constante,
        }
    else:
        objetivo_global, logrado_global, diagnostico_global = calcular_objetivo_v3(
            histograma_global, porcentaje, metodo, puntos, armonicos
        )
        imagen_v3_global, histograma_v3_global = reasignar_pixeles_por_frecuencia(imagen, objetivo_global)

    # Selección adaptativa: la búsqueda es 3×3, pero el procesamiento local usa 2×2.
    diagnostico_seleccion = seleccionar_cuadricula_adaptativa(
        imagen,
        paso_busqueda=paso,
        banda_minima=banda,
        region_minima=region_min,
        peso_exterior=peso_ext,
        suavizado=suavizado,
    )
    corte_y = int(diagnostico_seleccion["corte_horizontal_px"])
    corte_x = int(diagnostico_seleccion["corte_vertical_px"])
    filas = [np.arange(0, corte_y), np.arange(corte_y, alto)]
    columnas = [np.arange(0, corte_x), np.arange(corte_x, ancho)]
    centros_y = np.array([grupo.mean() for grupo in filas], dtype=float)
    centros_x = np.array([grupo.mean() for grupo in columnas], dtype=float)

    def pixeles_de_celdas(celdas):
        partes = []
        for i, j in sorted(celdas):
            y0, y1 = int(filas[i][0]), int(filas[i][-1]) + 1
            x0, x1 = int(columnas[j][0]), int(columnas[j][-1]) + 1
            partes.append(imagen[y0:y1, x0:x1].ravel())
        return partes[0] if len(partes) == 1 else np.concatenate(partes)

    def estadisticas_grupo(celdas):
        pixeles = pixeles_de_celdas(celdas)
        return {
            "desviacion": float(np.std(pixeles)),
            "media": float(np.mean(pixeles)),
            "pixeles": int(pixeles.size),
            "niveles": int(np.unique(pixeles).size),
        }

    def adyacentes(celdas_a, celdas_b):
        return any(abs(i-k) + abs(j-l) == 1 for i, j in celdas_a for k, l in celdas_b)

    grupos = [{"celdas": {(i, j)}} for i in range(2) for j in range(2)]
    for grupo in grupos:
        grupo.update(estadisticas_grupo(grupo["celdas"]))

    historial_fusiones = []
    respaldo_imagen_completa = False
    while fusionar:
        grupos_bajos = [idx for idx, grupo in enumerate(grupos) if grupo["desviacion"] < umbral]
        if not grupos_bajos:
            break
        if len(grupos) == 1:
            respaldo_imagen_completa = True
            break
        idx_bajo = min(grupos_bajos, key=lambda idx: (grupos[idx]["desviacion"], grupos[idx]["pixeles"]))
        grupo_bajo = grupos[idx_bajo]
        vecinos = [idx for idx, otro in enumerate(grupos)
                   if idx != idx_bajo and adyacentes(grupo_bajo["celdas"], otro["celdas"])]
        if not vecinos:
            grupo_completo = {"celdas": set().union(*(g["celdas"] for g in grupos))}
            grupo_completo.update(estadisticas_grupo(grupo_completo["celdas"]))
            grupos = [grupo_completo]
            respaldo_imagen_completa = True
            break
        idx_vecino = min(vecinos, key=lambda idx: (grupos[idx]["desviacion"], grupos[idx]["pixeles"]))
        vecino = grupos[idx_vecino]
        nuevas_celdas = set(grupo_bajo["celdas"]) | set(vecino["celdas"])
        nuevas_stats = estadisticas_grupo(nuevas_celdas)
        historial_fusiones.append({
            "celdas_antes": sorted(grupo_bajo["celdas"]),
            "desviacion_antes": grupo_bajo["desviacion"],
            "vecino_fusionado": sorted(vecino["celdas"]),
            "desviacion_vecino": vecino["desviacion"],
            "celdas_despues": sorted(nuevas_celdas),
            "desviacion_despues": nuevas_stats["desviacion"],
        })
        grupo_fusionado = {"celdas": nuevas_celdas, **nuevas_stats}
        for idx in sorted((idx_bajo, idx_vecino), reverse=True):
            grupos.pop(idx)
        grupos.append(grupo_fusionado)

    # Mapas en float32; las transformaciones siguen siendo finales de 8 bits.
    mapas_locales = np.empty((2, 2, 256), dtype=np.float32)
    frecuencias_objetivo_locales = np.zeros((2, 2, 256), dtype=np.int64)
    frecuencias_logradas_locales = np.zeros_like(frecuencias_objetivo_locales)
    cdfs_entrada_locales = np.zeros((2, 2, 256), dtype=float)
    cdfs_salida_locales = np.zeros_like(cdfs_entrada_locales)
    diagnosticos_locales = []
    mapas_detalle = {}
    grupo_por_celda = {}
    for numero_grupo, grupo in enumerate(grupos, start=1):
        celdas = set(grupo["celdas"])
        pixeles = pixeles_de_celdas(celdas)
        h = np.bincount(pixeles, minlength=256).astype(np.int64)
        objetivo, logrado, diag = calcular_objetivo_v3(h, porcentaje, metodo, puntos, armonicos)
        mapa, detalle_mapa = crear_mapa_intensidad(h, logrado, proteger_constante=proteger_constantes)
        cdf_entrada = np.cumsum(h) / h.sum()
        cdf_salida = np.cumsum(logrado) / logrado.sum()
        nombre_celdas = "; ".join(f"({i+1},{j+1})" for i, j in sorted(celdas))
        for i, j in celdas:
            mapas_locales[i, j] = mapa
            frecuencias_objetivo_locales[i, j] = objetivo
            frecuencias_logradas_locales[i, j] = logrado
            cdfs_entrada_locales[i, j] = cdf_entrada
            cdfs_salida_locales[i, j] = cdf_salida
            mapas_detalle[(i, j)] = detalle_mapa
            grupo_por_celda[(i, j)] = numero_grupo
            y0, y1 = int(filas[i][0]), int(filas[i][-1]) + 1
            x0, x1 = int(columnas[j][0]), int(columnas[j][-1]) + 1
            region_original = imagen[y0:y1, x0:x1]
            diagnosticos_locales.append({
                "grupo": numero_grupo,
                "celda_original": f"({i+1},{j+1})",
                "celdas_fusionadas": nombre_celdas,
                "y_inicio_celda": y0, "y_fin_exclusivo_celda": y1,
                "x_inicio_celda": x0, "x_fin_exclusivo_celda": x1,
                "desviacion_celda_original": float(np.std(region_original)),
                "desviacion_grupo_fusionado": grupo["desviacion"],
                "media_grupo_fusionado": grupo["media"],
                "pixeles_grupo": grupo["pixeles"],
                "niveles_grupo": grupo["niveles"],
                "r_min": diag["r_min"], "r_max": diag["r_max"],
                "porcentaje_intervalo_real": diag["porcentaje_real"],
                "A": diag["A"], "B": diag["B"], "C": diag["C"], "D": diag["D"],
                "niveles_ambiguos_con_varias_salidas": detalle_mapa["cantidad_ambiguas"],
                "region_constante_protegida": detalle_mapa["region_constante_protegida"],
                "fallback": diag.get("fallback", ""),
            })

    # Interpolación espacial bilineal de las tablas de transformación locales.
    idx_y0, idx_y1, peso_y = _indices_y_pesos_interpolacion(np.arange(alto), centros_y)
    idx_x0, idx_x1, peso_x = _indices_y_pesos_interpolacion(np.arange(ancho), centros_x)
    # La interpolación bilineal se procesa por bloques de filas para mantener
    # controlada la memoria en imágenes grandes, sin cambiar la regla espacial.
    interpolada = np.empty((alto, ancho), dtype=np.float32)
    wx = peso_x[None, :].astype(np.float32)
    tamano_bloque = 256
    for y_inicio in range(0, alto, tamano_bloque):
        y_fin = min(alto, y_inicio + tamano_bloque)
        intensidades_bloque = imagen[y_inicio:y_fin].astype(np.int64)
        T00 = mapas_locales[idx_y0[y_inicio:y_fin, None], idx_x0[None, :], intensidades_bloque]
        T01 = mapas_locales[idx_y0[y_inicio:y_fin, None], idx_x1[None, :], intensidades_bloque]
        T10 = mapas_locales[idx_y1[y_inicio:y_fin, None], idx_x0[None, :], intensidades_bloque]
        T11 = mapas_locales[idx_y1[y_inicio:y_fin, None], idx_x1[None, :], intensidades_bloque]
        wy = peso_y[y_inicio:y_fin, None].astype(np.float32)
        interpolada[y_inicio:y_fin] = (
            (1-wy)*(1-wx)*T00 + (1-wy)*wx*T01 + wy*(1-wx)*T10 + wy*wx*T11
        )
    imagen_v3_local = np.clip(np.rint(interpolada), 0, 255).astype(np.uint8)

    imagen_he = cv2.equalizeHist(imagen)
    clahe = cv2.createCLAHE(clipLimit=clahe_limite, tileGridSize=(clahe_grid, clahe_grid))
    imagen_clahe = clahe.apply(imagen)
    nombre_local = "V3 local + fusión adaptativa" if fusionar else "V3 local sin fusión"

    resultados_gris = {
        "Original": imagen,
        "V3 global": imagen_v3_global,
        nombre_local: imagen_v3_local,
        "HE": imagen_he,
        "CLAHE": imagen_clahe,
    }
    resultados_visuales = {
        "Original": rgb if es_color else imagen,
        "V3 global": convertir_a_rgb_con_luminancia(rgb, imagen_v3_global, es_color),
        nombre_local: convertir_a_rgb_con_luminancia(rgb, imagen_v3_local, es_color),
        "HE": convertir_a_rgb_con_luminancia(rgb, imagen_he, es_color),
        "CLAHE": convertir_a_rgb_con_luminancia(rgb, imagen_clahe, es_color),
    }
    tabla_metricas = pd.DataFrame({nombre: calcular_metricas(img) for nombre, img in resultados_gris.items()}).T
    tabla_diagnosticos_locales = pd.DataFrame(diagnosticos_locales)
    historial_df = pd.DataFrame(historial_fusiones)

    # El cuadrante con mayor gradiente sirve para inspeccionar histograma/CDF/T(r).
    gradiente = diagnostico_seleccion["magnitud_gradiente"]
    medias_cuadrantes = {}
    for i, ys in enumerate(filas):
        for j, xs in enumerate(columnas):
            bloque = gradiente[int(ys[0]):int(ys[-1])+1, int(xs[0]):int(xs[-1])+1]
            medias_cuadrantes[(i, j)] = float(np.mean(bloque))
    fila_ejemplo, columna_ejemplo = max(medias_cuadrantes, key=medias_cuadrantes.get)
    ys, xs = filas[fila_ejemplo], columnas[columna_ejemplo]
    region_ejemplo = imagen[int(ys[0]):int(ys[-1])+1, int(xs[0]):int(xs[-1])+1]
    hist_ejemplo = np.bincount(region_ejemplo.ravel(), minlength=256)

    columnas_mapas = {"intensidad_entrada": niveles}
    columnas_cdf = {"intensidad": niveles}
    for i in range(2):
        for j in range(2):
            columnas_mapas[f"T_region_{i+1}_{j+1}"] = mapas_locales[i, j]
            columnas_cdf[f"CDF_entrada_{i+1}_{j+1}"] = cdfs_entrada_locales[i, j]
            columnas_cdf[f"CDF_salida_{i+1}_{j+1}"] = cdfs_salida_locales[i, j]
    tabla_mapas = pd.DataFrame(columnas_mapas)
    tabla_cdfs = pd.DataFrame(columnas_cdf)

    return {
        "imagen_rgb_original": rgb,
        "imagen": imagen,
        "es_color": es_color,
        "resultados_gris": resultados_gris,
        "resultados_visuales": resultados_visuales,
        "tabla_metricas": tabla_metricas,
        "diagnostico_global": diagnostico_global,
        "diagnostico_seleccion": diagnostico_seleccion,
        "corte_x": corte_x,
        "corte_y": corte_y,
        "filas": filas,
        "columnas": columnas,
        "mapas_locales": mapas_locales,
        "cdfs_entrada_locales": cdfs_entrada_locales,
        "cdfs_salida_locales": cdfs_salida_locales,
        "tabla_diagnosticos_locales": tabla_diagnosticos_locales,
        "historial_fusiones": historial_df,
        "grupos": grupos,
        "respaldo_imagen_completa": respaldo_imagen_completa,
        "nombre_local": nombre_local,
        "fila_ejemplo": fila_ejemplo,
        "columna_ejemplo": columna_ejemplo,
        "hist_ejemplo": hist_ejemplo,
        "region_ejemplo": region_ejemplo,
        "tabla_mapas": tabla_mapas,
        "tabla_cdfs": tabla_cdfs,
        "parametros": p,
        "histograma_v3_global": histograma_v3_global,
        "objetivo_global": objetivo_global,
        "frecuencias_objetivo_locales": frecuencias_objetivo_locales,
        "frecuencias_logradas_locales": frecuencias_logradas_locales,
    }
