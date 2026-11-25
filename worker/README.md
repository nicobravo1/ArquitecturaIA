# Worker de explicabilidad SHAP (pendiente)

Consume eventos de RabbitMQ, calcula valores SHAP (`shap.TreeExplainer`) y guarda el
reporte de auditoría de cada transacción (PostgreSQL / DynamoDB). Fuera del camino crítico
de latencia (patrón CQRS: el comando decide, la consulta audita).
