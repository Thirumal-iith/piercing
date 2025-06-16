import pickle

with open("scenario.pkl", "rb") as f:
    scenario = pickle.load(f)

with open("strategy.pkl", "rb") as f:
    cl_strategy = pickle.load(f)

results = []
for experience in scenario.train_stream:
    print("Start of experience:", experience.current_experience)
    print("Current Classes:", experience.classes_in_this_experience)
    cl_strategy.train(experience)
    print("Training completed")
    print("Computing accuracy on the whole test set")
    results.append(cl_strategy.eval(scenario.test_stream))

with open("results.pkl", "wb") as f:
    pickle.dump(results, f)
