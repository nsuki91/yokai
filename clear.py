import pickle

num = {}
names = []
pickle.dump(num, open("data/number.pkl", "wb"))
pickle.dump(names, open("data/names.pkl", "wb"))
print("[INFO] Cleared!")
