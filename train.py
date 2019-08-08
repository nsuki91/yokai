from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

data = pickle.loads(open("data/embeddings.pickle", "rb").read())

le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("[INFO] starting...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open("data/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open("data/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
print("[INFO] done!")
