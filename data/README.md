# Dataset

**Credit Card Fraud Detection (mlg-ulb)** — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Coloca el archivo en `data/creditcard.csv` (no se versiona: ~150 MB).

```bash
pip install kaggle   # requiere ~/.kaggle/kaggle.json
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
```

Esquema esperado: `Time, V1..V28, Amount, Class` (284.807 filas, 492 fraudes).
El notebook verifica el esquema y registra el SHA-256 del archivo en `models/metadata.json` (lineage de datos).
