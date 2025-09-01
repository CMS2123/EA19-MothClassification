import argparse, json, os, pandas as pd, numpy as np, tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

p = argparse.ArgumentParser()
p.add_argument("--weights", required=True)
p.add_argument("--splits", required=True)   # folder with test.csv
p.add_argument("--classes", required=True)  # artifacts/class_index.json
p.add_argument("--img_size", type=int, default=224)
p.add_argument("--batch", type=int, default=32)
args = p.parse_args()

df = pd.read_csv(os.path.join(args.splits, "test.csv"))
class_index = json.load(open(args.classes))
inv = {v:k for k,v in class_index.items()}
paths = df["filepath"].tolist()
y_true = df["label"].map(class_index).values

def _load(p):
    x = tf.io.read_file(p); x = tf.image.decode_jpeg(x, 3)
    x = tf.image.resize(x, (args.img_size, args.img_size), antialias=True)
    return tf.cast(x, tf.float32)

ds = tf.data.Dataset.from_tensor_slices(paths).map(lambda p: (_load(p),), num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(args.batch).prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.load_model(args.weights)
probs = model.predict(ds, verbose=0)
y_pred = probs.argmax(1)

print("Top-1 accuracy:", round(accuracy_score(y_true, y_pred), 4))
print("\nClassification report (head):")
print(classification_report(y_true, y_pred, target_names=[inv[i] for i in range(len(inv))], digits=3)[:1200])
