### A transformer is a type of artificial intelligence model that learns to understand and generate human-like text by analyzing patterns in large amounts of text data.  Transformer is an Architecture uses self-attention to transform one whole sentence into a single sentence.  Transformers enable machines to understand, interpret, and generate human language in a way that's more accurate than ever before

```
column_transformer = ColumnTransformer([
    ('num', num_pipe, num_columns),
    ('cat', cat_pipe, cat_columns)
], remainder='drop')

cat_columns = X_train.columns[X_train.dtypes == "object"].tolist()
num_columns = X_train.columns[(X_train.dtypes == "int64") | (X_train.dtypes == "float64")].tolist()
```
