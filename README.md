# OVOSAURUS

classify language from spoken audio

OVOSAURUS turns an audio classification problem into a text classification problem, allowing for new models to be quickly trained on limited data for different language combinations on demand

1 - turn audio into a sequence of IPA phonemes, initial implementation uses Allosaurus, this step is lang agnostic (240 different phonemes)
2 - train a classic machine learning classifier on the phonemes


# Training

see scripts/gather_features.py for the code used to generate the dataset, a subset of VoxLingua107 was used

see train_svc.py for initial implementation using tfidf + SVC

see train_voter.py for initial implementation using a soft voting classifier (SVC, DecisionTree, LogisticRegression)

# Usage

not ready, open during construction, browse the code or something