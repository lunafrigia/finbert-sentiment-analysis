# Análisis de Sentimiento Financiero con FinBERT

Clasificación de frases de noticias financieras como **positivas**, **negativas** o **neutrales** utilizando FinBERT, un modelo Transformer pre-entrenado para el dominio financiero.

## Resultados

| Métrica | Score |
|---|---|
| Accuracy | 98.4% |
| F1 Macro | 98.0% |
| Errores en test | 3 / 190 |

## Dataset

**Financial PhraseBank** (Malo et al., 2014) — 1,264 frases de noticias financieras anotadas por 16 expertos. Se usa el subset `sentences_allagree` (consenso total entre anotadores) para maximizar la calidad de las etiquetas.

## Modelo

**FinBERT** (`ProsusAI/finbert`) — BERT fine-tuneado sobre comunicados financieros, reportes de ganancias y noticias de mercado. Partimos de este modelo pre-entrenado y hacemos fine-tuning sobre Financial PhraseBank.

## Pipeline

1. **EDA** — Distribución de clases, longitud de textos, wordclouds por sentimiento
2. **Preprocesamiento** — Split estratificado 70/15/15, tokenización con AutoTokenizer
3. **Fine-tuning** — WeightedTrainer con pesos de clase para desbalance, early stopping, warmup
4. **Evaluación** — Classification report, matriz de confusión, análisis de confianza
5. **Análisis de errores** — Patrones de confusión, errores con alta confianza
6. **Interpretabilidad** — Oclusión de tokens para identificar palabras influyentes
7. **Inferencia** — 12 noticias financieras reales nunca vistas por el modelo
8. **Comparación** — FinBERT pre-trained (zero-shot) vs fine-tuned

## Tecnologías

- Python 3.12
- PyTorch + CUDA (GPU)
- Hugging Face Transformers
- scikit-learn
- matplotlib / seaborn

## Cómo ejecutar

```bash
# Clonar el repositorio
git clone https://github.com/lunafrigia/finbert-sentiment-analysis.git
cd finbert-sentiment-analysis

# Instalar dependencias
pip install transformers datasets torch scikit-learn matplotlib seaborn wordcloud accelerate

# Abrir el notebook
jupyter notebook FinBERT_Sentiment_Analysis.ipynb
```

> **Nota:** Se recomienda GPU para el entrenamiento. En una RTX 5060 el fine-tuning toma ~32 segundos. En CPU puede tomar 10-15 minutos.

## Estructura del proyecto

```
├── FinBERT_Sentiment_Analysis.ipynb   # Notebook completo
├── README.md                          # Este archivo
└── finbert_sentiment_final/           # Modelo fine-tuneado (se genera al correr el notebook)
    ├── config.json
    ├── model.safetensors
    └── tokenizer.json
```

## Hallazgos clave

- El modelo aprende señales financieras reales: palabras como "increase", "profit", "declined" son las más influyentes en las predicciones
- Los pocos errores se concentran en la frontera neutral ↔ positivo, que es ambigua incluso para humanos
- Limitación interesante: "Fed announced unexpected rate hike" se clasifica como positiva, mostrando que el modelo prioriza el tono del texto sobre el impacto macroeconómico real

## Autor

**Mario Carvajal**
