# V3v5 — Aplicación Streamlit para mejoramiento de imágenes

## Qué reúne este proyecto

- **V3 (núcleo):** busca el intervalo de intensidades más corto que cubre el porcentaje configurado, aproxima la distribución mediante PCHIP, interpolación lineal o Fourier, ajusta el área del histograma objetivo al total de píxeles y reasigna los niveles de intensidad.
- **V3v2 (regiones locales):** construye una transformación de intensidad por región y combina espacialmente los mapas mediante interpolación bilineal.
- **V3v5 (selección adaptativa más reciente):** calcula la magnitud del gradiente en escala de grises, explora candidatas 3×3 y maximiza `J = gradiente medio central − λ × gradiente medio exterior`. La cuadrícula que se aplica a las regiones es 2×2 y sus ejes pasan por el punto medio de la región candidata elegida.
- **Fusión adaptativa:** permite fusionar regiones de baja desviación estándar con regiones vecinas y proteger regiones constantes.
- **Referencias:** HE (ecualización clásica) y CLAHE.

## Archivos

- `app.py`: interfaz de Streamlit.
- `modelo_v3.py`: algoritmos de histograma, selección de cuadrícula, mapas locales, fusión e interpolación.
- `requirements.txt`: dependencias para Streamlit Cloud.
- `referencia/`: cuaderno de Colab recibido como fuente de esta aplicación.

## Ejecutar localmente

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Streamlit Community Cloud

1. Descomprime el ZIP.
2. Sube `app.py`, `modelo_v3.py` y `requirements.txt` a la raíz de un repositorio de GitHub. Puedes incluir también `README.md` y la carpeta `referencia/`.
3. En Streamlit Community Cloud crea una app desde ese repositorio y selecciona `app.py` como archivo principal.
4. Streamlit instalará las dependencias de `requirements.txt`.

## Notas de interpretación

- El mapa de gradiente se calcula sobre la imagen en escala de grises; la transformación de intensidad también se calcula sobre esa representación.
- Salvaguarda para imágenes globalmente constantes: la V3 global conserva la imagen por identidad, porque no existe variación tonal que pueda redistribuirse sin introducir un cambio artificial.
- Para mostrar imágenes de entrada a color, la salida modifica el canal de luminancia `L` en el espacio Lab del original, siguiendo la estrategia del cuaderno.
- Las métricas de intensidad se calculan en escala de grises. El gradiente alto también puede corresponder a ruido o textura fina; el control de suavizado ayuda a moderar esa sensibilidad.
- La búsqueda exploratoria es 3×3, pero la partición final utilizada por la V3 local es 2×2.
- El archivo fuente adjunto se llamaba `Transformacion_Histograma_V3v4_Gradiente_2x2.ipynb`. La app se etiqueta **V3v5** siguiendo la identificación de versión indicada para este proyecto; no se ha alterado el cuaderno original incluido como referencia.
