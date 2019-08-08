# USAGE: python create.py -n number -i name
import pickle
import sys
import argparse

num = pickle.load(open("data/number.pkl", "rb"))
names = pickle.load(open("data/names.pkl", "rb"))

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number", required=True, help="")
ap.add_argument("-i", "--name", required=True, help="")
args = vars(ap.parse_args())


def save(num, names):
    pickle.dump(num, open("data/number.pkl", "wb"))
    pickle.dump(names, open("data/names.pkl", "wb"))
    print("Basariyla kaydedildi!")

def load():
    num = pickle.load(open("data/number.pkl", "rb"))
    names = pickle.load(open("data/names.pkl", "rb"))

def view():
	print(num)
	print(names)

name = args["name"]
number = args["number"]
num[name] = number
names.append(name)
save(num, names)
view()
