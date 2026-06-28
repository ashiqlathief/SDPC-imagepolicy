import pickle

with open("env_009_00004.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data)
