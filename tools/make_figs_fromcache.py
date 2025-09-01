import argparse, json, numpy as np, matplotlib.pyplot as plt, os
from sklearn.metrics import confusion_matrix

p = argparse.ArgumentParser()
p.add_argument("--probs", required=True)
p.add_argument("--labels", required=True)
p.add_argument("--classes", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
probs = np.load(args.probs)         # (N,K)
y_true = np.load(args.labels)       # (N,)
classes = json.load(open(args.classes))
K = len(classes)

y_pred = probs.argmax(1)
cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))
cm_norm = cm / cm.sum(1, keepdims=True)

plt.figure(figsize=(9,9))
plt.imshow(cm_norm, interpolation='nearest')
plt.title("Confusion (row-normalised)"); plt.colorbar()
plt.tight_layout(); plt.savefig(os.path.join(args.out, "confusion_base.png"), dpi=200)
print("Saved:", os.path.join(args.out, "confusion_base.png"))

