import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# fix a random seed for reproducability (stocastic)
seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv("Iris_Test.csv", header=None)
dataset = dataframe.values
x = dataset[:, 0:4].astype(float)
y = dataset[:, 4]

# One hot encoding the different iris flowers as arbitrary numbers

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

dummy_y = np_utils.to_categorical(encoded_y)

# defining model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

# evaluating model
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, x, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

