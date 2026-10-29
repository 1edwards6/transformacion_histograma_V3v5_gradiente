# -*- coding: utf-8 -*-
"""Aplicación Streamlit — V3v5: histograma expandido, regiones locales y gradiente."""
from __future__ import annotations
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle
from PIL import Image

from modelo_v3 import procesar_imagen_v3v5

st.set_page_config(page_title="V3v5 · Mejoramiento de imágenes", page_icon="🖼️", layout="wide")

st.title("V3v5 · Mejoramiento matemático de imágenes")
st.markdown(
    "La **V3** es el núcleo: construye y aproxima el histograma objetivo para redistribuir las intensidades. "
    "La **V3v2** incorpora transformaciones por regiones locales e interpolación espacial. "
    "La versión actual combina esas ideas con una **cuadrícula adaptable por gradiente**: explora una partición 3×3 "
    "para localizar una región con cambios rápidos de intensidad y coloca los dos ejes finales 2×2 en su punto medio."
)

with st.sidebar:
    st.header("Imagen")
    archivo = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"])
    st.divider()
    st.header("Modelo V3: histograma")
    porcentaje = st.slider("Porcentaje cubierto por el intervalo mínimo", 80.0, 100.0, 97.5, 0.5,
                           help="La V3 encuentra el intervalo de intensidades más corto que cubre este porcentaje de píxeles.")
    metodo = st.selectbox("Aproximación del histograma", ["PCHIP", "LINEAL", "FOURIER"], index=0)
    puntos = st.slider("Puntos de aproximación (PCHIP/LINEAL)", 2, 256, 150, 1)
    armonicos = st.slider("Número de armónicos (FOURIER)", 0, 64, 12, 1)
    st.caption("Los controles de puntos y armónicos solo afectan al método correspondiente.")
    st.divider()
    st.header("Cuadrícula adaptable por gradiente")
    paso = st.select_slider("Paso de búsqueda (%)", options=[2.5, 5.0, 7.5, 10.0], value=5.0)
    banda = st.slider("Banda exterior mínima candidata (%)", 5.0, 35.0, 10.0, 2.5,
                      help="Se ajusta automáticamente si contradice el ancho mínimo solicitado para la región central.")
    region_min = st.slider("Ancho/alto mínimo de región central (%)", 15.0, 70.0, 30.0, 5.0)
    peso_ext = st.slider("Peso de penalización del gradiente exterior", 0.0, 1.5, 0.55, 0.05,
                         help="Se maximiza J = gradiente central − peso × gradiente medio exterior.")
    suavizado = st.slider("Suavizado previo σ", 0.0, 5.0, 2.0, 0.25,
                          help="Un valor mayor reduce ruido fino antes de calcular las derivadas.")
    st.divider()
    st.header("Regiones locales")
    fusionar = st.checkbox("Fusionar celdas con baja dispersión", value=True)
    umbral = st.slider("Umbral de desviación estándar para fusionar", 0.0, 60.0, 10.0, 0.5,
                       disabled=not fusionar)
    proteger = st.checkbox("Proteger regiones constantes", value=True,
                           help="Si una región contiene un único nivel de intensidad, se conserva sin realzarla artificialmente.")
    st.divider()
    st.header("Método de referencia CLAHE")
    clahe_limite = st.slider("Límite de contraste CLAHE", 0.5, 5.0, 2.0, 0.25)
    clahe_grid = st.slider("Cuadrícula CLAHE", 2, 16, 8, 1)

if archivo is None:
    st.info("Carga una imagen desde el panel lateral para ejecutar la V3v5.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("V3 · Histograma")
        st.write("Busca el intervalo mínimo que contiene el porcentaje seleccionado, aproxima su distribución y construye un histograma objetivo con área ajustada al total de píxeles.")
    with c2:
        st.subheader("V3v2 · Regiones locales")
        st.write("Calcula mapas de transformación por región y combina sus efectos mediante interpolación espacial bilineal para reducir fronteras bruscas.")
    with c3:
        st.subheader("V3v5 · Gradiente adaptable")
        st.write("Compara el gradiente medio de la zona central candidata con el de sus ocho regiones exteriores y usa el centro detectado para establecer la partición final 2×2.")
    st.stop()

image_bytes = archivo.getvalue()
params = {
    "porcentaje_conservado": porcentaje,
    "metodo": metodo,
    "num_puntos": puntos,
    "num_armonicos": armonicos,
    "paso_busqueda": paso,
    "banda_minima": banda,
    "region_minima": region_min,
    "peso_exterior": peso_ext,
    "suavizado_gradiente": suavizado,
    "fusionar_celdas": fusionar,
    "umbral_desviacion": umbral,
    "proteger_constantes": proteger,
    "clahe_limite": clahe_limite,
    "clahe_grid": clahe_grid,
}

@st.cache_data(show_spinner=False, max_entries=3)
def _procesar_cache(datos_imagen: bytes, parametros_serializados: tuple):
    imagen_pil = Image.open(io.BytesIO(datos_imagen)).convert("RGB")
    rgb = np.asarray(imagen_pil, dtype=np.uint8)
    nombres = [
        "porcentaje_conservado", "metodo", "num_puntos", "num_armonicos", "paso_busqueda",
        "banda_minima", "region_minima", "peso_exterior", "suavizado_gradiente",
        "fusionar_celdas", "umbral_desviacion", "proteger_constantes", "clahe_limite", "clahe_grid"
    ]
    p = dict(zip(nombres, parametros_serializados))
    return procesar_imagen_v3v5(rgb, p)

params_tuple = tuple(params[k] for k in [
    "porcentaje_conservado", "metodo", "num_puntos", "num_armonicos", "paso_busqueda",
    "banda_minima", "region_minima", "peso_exterior", "suavizado_gradiente",
    "fusionar_celdas", "umbral_desviacion", "proteger_constantes", "clahe_limite", "clahe_grid"
])

try:
    with st.spinner("Procesando histograma, buscando la cuadrícula y calculando las regiones locales…"):
        R = _procesar_cache(image_bytes, params_tuple)
except Exception as exc:
    st.error(f"No fue posible procesar la imagen: {exc}")
    st.exception(exc)
    st.stop()

D = R["diagnostico_seleccion"]
region = D["region_candidata_3x3"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Gradiente central candidato", f"{D['gradiente_medio_central']:.3f}")
m2.metric("Gradiente medio exterior", f"{D['gradiente_medio_exterior']:.3f}")
m3.metric("Puntuación J", f"{D['puntaje']:.3f}")
m4.metric("Grupos locales finales", str(len(R["grupos"])))
if abs(D["banda_minima_efectiva_pct"] - banda) > 1e-6:
    st.info(f"Para permitir una región central de al menos {region_min:.1f} %, la banda exterior mínima efectiva se ajustó de {banda:.1f} % a {D['banda_minima_efectiva_pct']:.1f} %.")

# Helpers for plots and downloads
def _png_bytes(array):
    buf = io.BytesIO()
    arr = np.asarray(array)
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()

def _csv_bytes(df, index=True):
    return df.to_csv(index=index).encode("utf-8-sig")

def _make_download_bundle():
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imagen_original.png", _png_bytes(R["resultados_visuales"]["Original"]))
        for name, image in R["resultados_visuales"].items():
            if name == "Original":
                continue
            safe = name.lower().replace(" ", "_").replace("+", "mas").replace("/", "_")
            zf.writestr(f"{safe}.png", _png_bytes(image))
        zf.writestr("metricas_comparacion.csv", _csv_bytes(R["tabla_metricas"]))
        zf.writestr("diagnostico_regiones_locales.csv", _csv_bytes(R["tabla_diagnosticos_locales"], index=False))
        zf.writestr("historial_fusiones.csv", _csv_bytes(R["historial_fusiones"], index=False))
        zf.writestr("mapas_transformacion_locales.csv", _csv_bytes(R["tabla_mapas"], index=False))
        zf.writestr("cdfs_por_region.csv", _csv_bytes(R["tabla_cdfs"], index=False))
        zf.writestr("gradientes_cuadricula_candidata_3x3.csv", _csv_bytes(D["tabla_gradientes_3x3"], index=False))
    memory.seek(0)
    return memory.getvalue()

tab_comparacion, tab_grid, tab_hist, tab_local, tab_download = st.tabs([
    "Comparación visual", "Cuadrícula y gradiente", "Histogramas y CDF", "Transformación local", "Diagnóstico y descargas"
])

with tab_comparacion:
    st.subheader("Resultados de los cinco métodos")
    names = list(R["resultados_visuales"].keys())
    cols = st.columns(len(names))
    for col, name in zip(cols, names):
        with col:
            st.image(R["resultados_visuales"][name], caption=name, use_container_width=True)
    st.caption("Las métricas se calculan sobre la intensidad en escala de grises. En imágenes a color, las transformaciones se visualizan modificando el canal de luminancia L en Lab, como en el cuaderno fuente.")
    st.subheader("Comparación Original y V3 local")
    a, b = st.columns(2)
    a.image(R["resultados_visuales"]["Original"], caption="Original", use_container_width=True)
    b.image(R["resultados_visuales"][R["nombre_local"]], caption=R["nombre_local"], use_container_width=True)
    st.dataframe(R["tabla_metricas"].round(4), use_container_width=True)

with tab_grid:
    st.subheader("Cuadrícula final 2×2 sobre la imagen")
    st.write(f"Región candidata de la búsqueda exploratoria 3×3: x = {region['x1_pct']:.1f}–{region['x2_pct']:.1f} % del ancho; y = {region['y1_pct']:.1f}–{region['y2_pct']:.1f} % del alto.")
    st.write(f"Eje vertical final: x = {R['corte_x']} px ({D['corte_vertical_pct']:.2f} %). Eje horizontal final: y = {R['corte_y']} px ({D['corte_horizontal_pct']:.2f} %).")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(R["imagen"], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Región candidata y ejes finales 2×2")
    axes[0].axis("off")
    axes[0].add_patch(Rectangle((region["x1_px"]-0.5, region["y1_px"]-0.5), region["x2_px"]-region["x1_px"], region["y2_px"]-region["y1_px"], fill=False, edgecolor="red", linewidth=2, linestyle="--"))
    axes[0].axvline(R["corte_x"]-0.5, color="navy", linewidth=2)
    axes[0].axhline(R["corte_y"]-0.5, color="navy", linewidth=2)
    axes[1].imshow(D["magnitud_gradiente"], cmap="inferno")
    axes[1].set_title("Magnitud del gradiente")
    axes[1].axis("off")
    axes[1].axvline(R["corte_x"]-0.5, color="cyan", linewidth=1.5)
    axes[1].axhline(R["corte_y"]-0.5, color="cyan", linewidth=1.5)
    axes[1].add_patch(Rectangle((region["x1_px"]-0.5, region["y1_px"]-0.5), region["x2_px"]-region["x1_px"], region["y2_px"]-region["y1_px"], fill=False, edgecolor="white", linewidth=1.5, linestyle="--"))
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("**Criterio de selección**")
    st.latex(r"J=\overline{G}_{\mathrm{central}}-\lambda\,\overline{G}_{\mathrm{exterior}}")
    st.write("Se elige la candidata con mayor puntuación J. Los ejes usados por las transformaciones locales son el punto medio geométrico de esa región candidata.")
    st.subheader("Gradiente medio por celda en la candidata 3×3")
    st.dataframe(D["tabla_gradientes_3x3"].round(4), use_container_width=True, hide_index=True)

with tab_hist:
    st.subheader("Histogramas normalizados")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, image in R["resultados_gris"].items():
        hist = np.bincount(image.ravel(), minlength=256)
        ax.plot(np.arange(256), hist / image.size, label=name, linewidth=1.3)
    ax.set(xlabel="Intensidad", ylabel="Frecuencia relativa", xlim=(0, 255), title="Distribución de intensidades")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.subheader("Funciones de distribución acumulada (CDF)")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, image in R["resultados_gris"].items():
        hist = np.bincount(image.ravel(), minlength=256)
        ax.plot(np.arange(256), np.cumsum(hist) / image.size, label=name, linewidth=1.4)
    ax.set(xlabel="Intensidad", ylabel="Probabilidad acumulada", xlim=(0, 255), ylim=(-0.02, 1.02), title="CDF de cada resultado")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab_local:
    i, j = R["fila_ejemplo"], R["columna_ejemplo"]
    st.subheader(f"Análisis del cuadrante con mayor gradiente: fila {i+1}, columna {j+1}")
    st.image(R["region_ejemplo"], caption="Región de entrada seleccionada", clamp=True, use_container_width=True)
    h = R["hist_ejemplo"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].plot(np.arange(256), h, linewidth=1.4)
    axes[0].set(title="Histograma local", xlabel="Intensidad", ylabel="Frecuencia")
    axes[0].grid(alpha=0.2)
    axes[1].plot(np.arange(256), R["cdfs_entrada_locales"][i, j], label="CDF de entrada")
    axes[1].plot(np.arange(256), R["cdfs_salida_locales"][i, j], label="CDF de salida V3")
    axes[1].set(title="CDF de la región", xlabel="Intensidad", ylabel="Probabilidad", ylim=(-0.02, 1.02))
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    axes[2].plot(np.arange(256), R["mapas_locales"][i, j], label="T(r)")
    axes[2].plot(np.arange(256), np.arange(256), linestyle="--", alpha=0.55, label="Identidad")
    axes[2].set(title="Transformación local", xlabel="Intensidad de entrada", ylabel="Intensidad de salida", xlim=(0, 255), ylim=(0, 255))
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("La curva T(r) se construye a partir de la distribución local de frecuencias de la región y de la reasignación ordenada de píxeles de la V3. Para las regiones constantes, la opción de protección conserva el comportamiento de identidad.")

with tab_download:
    st.subheader("Métricas globales")
    st.dataframe(R["tabla_metricas"].round(4), use_container_width=True)
    st.subheader("Diagnóstico de regiones locales")
    st.dataframe(R["tabla_diagnosticos_locales"].round(4), use_container_width=True, hide_index=True)
    if not R["historial_fusiones"].empty:
        st.subheader("Historial de fusiones adaptativas")
        st.dataframe(R["historial_fusiones"], use_container_width=True, hide_index=True)
    st.write(f"Fusión adaptativa: {'activada' if fusionar else 'desactivada'}. Grupos finales: {len(R['grupos'])}. Respaldo de imagen completa: {'sí' if R['respaldo_imagen_completa'] else 'no'}.")
    st.write(f"V3 global — intervalo [{R['diagnostico_global']['r_min']}, {R['diagnostico_global']['r_max']}], A={R['diagnostico_global']['A']:.5g}, B={R['diagnostico_global']['B']:.5g}, C={R['diagnostico_global']['C']:.5g}, D={R['diagnostico_global']['D']:.5g}.")
    st.download_button("Descargar resultados + CSV en ZIP", data=_make_download_bundle(), file_name="resultados_V3v5_gradiente_2x2.zip", mime="application/zip", use_container_width=True)
    st.download_button("Descargar imagen V3 local", data=_png_bytes(R["resultados_visuales"][R["nombre_local"]]), file_name="imagen_V3v5_local.png", mime="image/png", use_container_width=True)
    st.download_button("Descargar métricas CSV", data=_csv_bytes(R["tabla_metricas"]), file_name="metricas_V3v5.csv", mime="text/csv", use_container_width=True)

st.divider()
st.caption("V3v5 · Núcleo de transformación del histograma V3 + mapas locales V3v2 + cuadrícula 2×2 seleccionada mediante gradiente. HE y CLAHE se incluyen como referencias de comparación.")
