import pickle
with open('./steel_data_model.pkl', 'rb') as f:
    model = pickle.load(f)  # this is your XGBClassifier
print(model)