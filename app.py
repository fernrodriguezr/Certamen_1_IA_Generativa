"""
SolarCheck IA – Diagnóstico visual de paneles solares
App Gradio: imagen -> EfficientNet-B0 (clase + confianza) -> gráfico -> LLM (Gemini) -> recomendación
"""
import os, json, glob
import torch, timm
import gradio as gr
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------
# 1) Cargar modelo entrenado y metadatos (clases + métricas de test)
# ---------------------------------------------------------------
with open("modelo/clases.json") as f:
    META = json.load(f)
CLASES = META["clases"]
TEST_ACC = META["test_accuracy"]
TEST_BACC = META["test_balanced_accuracy"]

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASES))
model.load_state_dict(torch.load("modelo/modelo.pt", map_location="cpu"))
model.eval()

# Mismas transformaciones que en evaluación (notebook, paso 6)
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Nombres en español para mostrar al usuario
NOMBRES = {
    "Bird-drop": "Excremento de aves",
    "Clean": "Limpio",
    "Dusty": "Polvo acumulado",
    "Electrical-damage": "Daño eléctrico",
    "Physical-Damage": "Daño físico",
    "Snow-Covered": "Cubierto de nieve",
}
UMBRAL_CONFIANZA = 0.60  # bajo este valor se pide revisión humana


# ---------------------------------------------------------------
# 2) Clasificación
# ---------------------------------------------------------------
def clasificar(img):
    if img is None:
        return None, "⚠️ Sube una imagen primero.", None
    x = TF(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = model(x).softmax(dim=1)[0]
    probs_dict = {NOMBRES[c]: float(p) for c, p in zip(CLASES, probs)}
    idx = int(probs.argmax())
    clase, conf = CLASES[idx], float(probs[idx])

    texto = f"### Resultado: **{NOMBRES[clase]}**\nConfianza de esta predicción: **{conf:.1%}**"
    if conf < UMBRAL_CONFIANZA:
        texto += ("\n\n> ⚠️ **Confianza baja.** El modelo no está seguro; "
                  "se recomienda inspección visual por un técnico.")
    estado = {"clase": clase, "nombre": NOMBRES[clase], "confianza": conf}
    return probs_dict, texto, estado


# ---------------------------------------------------------------
# 3) Recomendación con IA generativa (solo se envía TEXTO, no la imagen)
# ---------------------------------------------------------------
PROMPT_SISTEMA = """Eres un asistente técnico de operación y mantenimiento (O&M) de plantas fotovoltaicas.
Reglas:
- Responde SOLO sobre mantenimiento y operación de paneles solares. Si te preguntan otra cosa, indícalo.
- Responde en español, en máximo 120 palabras, con: (1) qué significa el hallazgo, (2) impacto probable en la generación, (3) 2-3 acciones recomendadas con prioridad (alta/media/baja).
- Si la confianza es menor a 60%, dilo al inicio y recomienda inspección visual por un técnico antes de actuar.
- Ante daño eléctrico o físico, advierte riesgo eléctrico: no manipular el panel, aislar el string y llamar a personal calificado.
- No inventes cifras exactas de pérdida; usa rangos generales y aclara que dependen de cada instalación.
- Termina recordando que es una herramienta de apoyo educativa, no un diagnóstico técnico certificado."""


def recomendar(estado):
    if not estado:
        return "Primero clasifica una imagen."
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ No se encontró la clave GEMINI_API_KEY (configúrala como *secret* del Space)."
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        mensaje = (f"{PROMPT_SISTEMA}\n\nResultado del clasificador de imágenes:\n"
                   f"- Condición detectada: {estado['nombre']}\n"
                   f"- Confianza del modelo: {estado['confianza']:.1%}\n"
                   f"Entrega la recomendación.")
        resp = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=mensaje)
        return resp.text
    except Exception as e:
        return f"⚠️ No se pudo generar la recomendación: {e}"


# ---------------------------------------------------------------
# 4) Interfaz
# ---------------------------------------------------------------
tema = gr.themes.Soft(primary_hue="amber", secondary_hue="blue", neutral_hue="slate")
ejemplos = sorted(glob.glob("ejemplos/*"))

# Compatibilidad: en Gradio 4-5 el tema va en Blocks; en Gradio 6+ va en launch()
import inspect
GRADIO_NUEVO = "theme" in inspect.signature(gr.Blocks.launch).parameters
blocks_kwargs = {"title": "SolarCheck IA"}
if not GRADIO_NUEVO:
    blocks_kwargs["theme"] = tema

with gr.Blocks(**blocks_kwargs) as demo:
    gr.Markdown(
        "# ☀️ SolarCheck IA\n"
        "### Diagnóstico visual del estado de paneles solares\n"
        "Sube una foto de un panel y la app detecta si está **limpio**, con **polvo**, "
        "**excremento de aves**, **nieve**, **daño eléctrico** o **daño físico**, "
        "y genera una recomendación de mantenimiento con IA.")

    estado = gr.State()

    with gr.Tabs():
        with gr.Tab("🔍 Diagnóstico"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(type="pil", label="Foto del panel",
                                      sources=["upload", "webcam"], height=320)
                    btn_clf = gr.Button("Analizar panel", variant="primary")
                    if ejemplos:
                        gr.Examples(ejemplos, inputs=img_in, label="Imágenes de ejemplo")
                with gr.Column(scale=1):
                    res_md = gr.Markdown()
                    probs_out = gr.Label(num_top_classes=6, label="Probabilidad por clase")
                    btn_llm = gr.Button("💡 Generar recomendación con IA", variant="secondary")
                    rec_out = gr.Markdown()

        with gr.Tab("📊 Desempeño del modelo"):
            gr.Markdown(
                f"**Modelo:** EfficientNet-B0 preentrenado en ImageNet, ajustado (*fine-tuning*) "
                f"con 608 imágenes.\n\n"
                f"| Métrica en conjunto de prueba (131 imágenes) | Valor |\n|---|---|\n"
                f"| Accuracy | **{TEST_ACC:.1%}** |\n"
                f"| Balanced accuracy | **{TEST_BACC:.1%}** |\n\n"
                "Estas métricas describen el desempeño **global** del modelo en datos que no vio "
                "al entrenar. Son distintas de la *confianza* de cada predicción, que indica qué tan "
                "seguro está el modelo sobre una imagen en particular.\n\n"
                "**Limitación principal:** la clase *Polvo acumulado* es la más difícil "
                "(se confunde con *Limpio*).")
            if os.path.exists("modelo/comparacion.png"):
                gr.Image("modelo/comparacion.png", label="Recall por clase: CLIP vs EfficientNet")

        with gr.Tab("ℹ️ Acerca de"):
            gr.Markdown(
                "**Datos:** *Solar Panel Images Clean and Faulty Images* (Kaggle, autor: Afroz). "
                "Licencia no especificada; uso exclusivamente académico.\n\n"
                "**Proyecto:** Electivo IA Generativa – Universidad del Desarrollo.\n\n"
                "**IA generativa:** la recomendación se genera con Google Gemini; "
                "solo se envía el resultado en texto (clase y confianza), nunca la imagen.")

    gr.Markdown(
        "---\n⚠️ *Prototipo educativo. Los resultados no reemplazan la inspección de un técnico "
        "calificado. Ante daño eléctrico, no manipule el panel.*")

    btn_clf.click(clasificar, inputs=img_in, outputs=[probs_out, res_md, estado])
    btn_llm.click(recomendar, inputs=estado, outputs=rec_out)

if __name__ == "__main__":
    demo.launch(theme=tema) if GRADIO_NUEVO else demo.launch()