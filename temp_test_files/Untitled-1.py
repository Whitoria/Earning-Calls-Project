import pickle
with open('motley-fool-data.pkl', 'rb') as f:
    data = pickle.load(f)

print(type(data))