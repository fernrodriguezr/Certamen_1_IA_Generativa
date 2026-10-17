# ☀️ SolarCheck IA – Diagnóstico visual de paneles solares

App web que clasifica el estado de un panel solar a partir de una foto y genera una recomendación de mantenimiento con IA generativa.

**Flujo:** 📷 Imagen → 🧠 EfficientNet-B0 (clase + confianza) → 📊 Probabilidades → 💬 Gemini → ✅ Recomendación

🔗 **App publicada:** _[pegar link de Hugging Face Spaces]_

---

## 1. Problema y motivación
La acumulación de polvo, nieve o excremento de aves sobre los paneles reduce la energía generada, y los daños eléctricos o físicos implican riesgo y pérdida de módulos. Inspeccionar visualmente miles de paneles es lento y costoso. SolarCheck IA apoya al personal de operación y mantenimiento (O&M) de plantas fotovoltaicas priorizando qué paneles limpiar o revisar.

## 2. Datos
| Campo | Detalle |
|---|---|
| Dataset | *Solar Panel Images Clean and Faulty Images* |
| Autor | Afroz (Kaggle: `pythonafroz`) |
| Enlace | https://www.kaggle.com/datasets/pythonafroz/solar-panel-images |
| Licencia | No especificada ("Unknown") – uso exclusivamente académico |
| Clases (6) | Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered |
| Clase excluida | `New` (solo 16 imágenes, insuficiente para entrenar) |

**Imágenes por clase y división (estratificada 70 / 15 / 15, semilla 42):**

| Clase | Train | Val | Test | Total |
|---|---|---|---|---|
| Bird-drop | 134 | 28 | 29 | 191 |
| Clean | 135 | 29 | 29 | 193 |
| Dusty | 133 | 28 | 29 | 190 |
| Electrical-damage | 72 | 16 | 15 | 103 |
| Physical-Damage | 48 | 11 | 10 | 69 |
| Snow-Covered | 86 | 18 | 19 | 123 |
| **Total** | **608** | **130** | **131** | **869** |

**Limitaciones:** licencia y origen de las fotos no documentados; desbalance 2,8:1 (Clean vs Physical-Damage); pocas imágenes en total; fotos heterogéneas (ángulo, iluminación, resolución); el test de Physical-Damage tiene solo 10 imágenes, por lo que su recall es poco estable. El dataset no se sube al repositorio: se descarga desde el enlace.

## 3. Modelo y resultados
Se compararon dos enfoques sobre el **mismo conjunto de prueba (131 imágenes)**:

| Métrica (test) | CLIP zero-shot (línea base) | **EfficientNet-B0 fine-tuned** |
|---|---|---|
| Accuracy | 0,481 | **0,740** |
| Balanced accuracy | 0,514 | **0,768** |

- **EfficientNet-B0** (timm, preentrenado en ImageNet) ajustado 15 épocas con AdamW, *data augmentation* y pesos por clase para compensar el desbalance. Se guardó la época con mejor balanced accuracy en validación.
- **Error más costoso:** clasificar un panel con polvo como limpio. El panel no se limpia y sigue perdiendo generación sin que nadie lo note. CLIP cometió este error en 14 de 29 casos; EfficientNet en 5 de 29.
- **Clase más difícil:** Dusty (recall 0,48), que se confunde con Clean y Physical-Damage.

![Comparación](modelo/comparacion.png)

## 4. App
Hecha con **Gradio** y publicada en **Hugging Face Spaces**. Incluye carga de imagen (archivo o cámara), ejemplos, clase predicha, confianza, probabilidades de todas las clases, métricas del modelo en test, recomendación con IA y advertencias (aviso educativo y revisión humana si la confianza es menor a 60%).

**IA generativa:** Google Gemini (`gemini-2.5-flash`). Solo se envía **texto** (clase y confianza), nunca la imagen. El *prompt* limita el tema al mantenimiento de paneles, pide revisión humana con confianza baja y advierte riesgo eléctrico ante daños. La clave se guarda como *secret* `GEMINI_API_KEY` en el Space y **no está en este repositorio**.

## 5. Estructura
```
├── app.py                  # aplicación Gradio
├── requirements.txt        # librerías
├── README.md
├── notebooks/
│   └── entrenamiento.ipynb # datos, CLIP vs EfficientNet, métricas
├── modelo/
│   ├── modelo.pt           # pesos EfficientNet-B0
│   ├── clases.json         # clases y métricas de test
│   └── comparacion.png
└── ejemplos/               # imágenes de prueba
```

## 6. Cómo ejecutar localmente
```bash
pip install -r requirements.txt
set GEMINI_API_KEY=tu_clave        # Windows  (Mac/Linux: export GEMINI_API_KEY=tu_clave)
python app.py
```

## 7. Integrantes
- Bejamin Araya
- Guillermo Barañado
- Fernando Rodríguez

## 8. Declaración de uso de IA
- **Claude (Anthropic):** apoyo para seleccionar el dataset, generar y explicar el código base del notebook (división de datos, CLIP, fine-tuning) y de `app.py`, y redactar este README. El grupo adaptó el código al problema, ejecutó el entrenamiento, interpretó los resultados y revisó cada parte.
- **Google Gemini:** genera las recomendaciones dentro de la app.

## 9. Consideraciones éticas
Prototipo educativo. No reemplaza la inspección de un técnico calificado. Ante daño eléctrico no se debe manipular el panel. El modelo se entrenó con una sola fuente de imágenes y puede fallar con otros tipos de panel, ángulos o condiciones de luz.
