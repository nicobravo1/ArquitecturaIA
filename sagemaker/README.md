# SageMaker (pendiente)

Despliegue del modelo en AWS SageMaker con el contenedor XGBoost integrado:

- `deploy.py`: sube `models/model.tar.gz` a S3, crea `Model`, `EndpointConfig` y `Endpoint`.
- Auto-scaling (Application Auto Scaling) por `SageMakerVariantInvocationsPerInstance`.
- `invoke_test.py`: prueba con `models/sample_payloads.json` (content-type `text/csv`).
