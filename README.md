# 🧠 Sistema de Reconocimiento Facial  
### Pipeline Batch + Streaming + Kafka + Spark + LBPH

Notas importantes El dataset no se encuentra subido, debido a el peso de este mismo y dado a que github tiene un limitación de peso de los archivos, se decidio no agregarlo en este repositorio , por ende
 si desea probar este proyecto debe crear una carpeta llamada "dataset" y dentro de esta subcarpetas, mediante el archivo "spark.ingest.py" va a analizar dicha dataset nueva.
 En caso de que no tenga un dataset propio, puede crearlo con el archivo "extract_frames.py"  , que permite la obtención de frames de videos subidos y automaticamente colocarlos en subcarpetas previamente
 creadas en la carpeta de dataset
 ---
 Comando de ejecución de este código:
``` python -m src.extract_frames```

## 1. Descripción del Proyecto
Este proyecto implementa un sistema completo de reconocimiento facial en tiempo real usando:

- **OpenCV LBPH** para reconocimiento facial  
- **Spark (Batch)** para procesar el dataset y generar el warehouse  
- **Kafka (Streaming)** para enviar eventos en tiempo real  
- **Spark Streaming** como consumidor distribuido  
- **Logging con Hash-Chain** para trazabilidad  
- **Métricas automáticas + visualizaciones**

El sistema funciona bajo una arquitectura híbrida **Batch + Streaming** que simula entornos de producción.

---

## 2. Instalación del Entorno Virtual

### Crear entorno virtual
```bash```
python -m venv .venv
Activar entorno
Windows PowerShell

```bash```
Copiar código
.\.venv\Scripts\Activate.ps1
Windows CMD

3. Instalación de Dependencias

```pip install -r requirements.txt```

---

4. Estructura del Proyecto

<img width="320" height="499" alt="image" src="https://github.com/user-attachments/assets/23614ed7-0dd6-4544-9086-e13cc10d3c98" />



5. Ejecución del Pipeline
   ---

5.1 Procesamiento Batch (Spark)

```python src/spark_ingest.py```
Genera:

bash
Copiar código
warehouse/faces.parquet
5.2 Entrenamiento del Modelo LBPH


```python -m src.train_lbph```
Genera:
- models/lbph_model.xml
- models/labels.json
 ---
 
5.3 Métricas y Gráficos

```python -m src.metricas```
Genera:

- metricas_train_test.csv
- metricas_cross_validation.csv
  ```python metricas_resultados.py```
  Crea:
  - accuracy_comparacion.png
  - curva_loss_accuracy.png
  - fps_folds.png
  - latencia_folds.png
  --- 
6. Ejecución del Sistema en Tiempo Real
   ---
   
6.1 Iniciar Apache Kafka
```bin/zookeeper-server-start.sh config/zookeeper.properties```
---
6.2 Iniciar el Servidor de Kafka (Broker)
   ```bin/kafka-server-start.sh config/server.properties```
