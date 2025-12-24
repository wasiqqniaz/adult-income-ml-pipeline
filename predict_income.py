
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# helper functions

def simplify_native_country(x):
    if x == 'United-States':
        return 'United-States'
    elif x == '?':
        return 'unknown'
    else:
        return 'Others'

def simplify_workclass(x):
    x = x.strip().lower()
    unemployed = ['without-pay', 'never-worked']
    employed = ['private', 'self-emp-not-inc', 'local-gov', 'state-gov', 'self-emp-inc', 'federal-gov']
    if x in unemployed:
        return 'Unemployed'
    elif x in employed:
        return 'Employed'
    else:
        return 'unknown'

def simplify_marital(x):
    if x in ['Married-civ-spouse', 'Married-AF-spouse']:
        return 'Married'
    elif x == 'Never-married':
        return 'Never-married'
    else:
        return 'Previously-married'

def simplify_occupation(x):
    white_collar = ['Armed-Forces','Prof-specialty', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Tech-support']
    blue_collar  = ['Craft-repair', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing']
    service      = ['Other-service', 'Protective-serv', 'Priv-house-serv']
    if x in white_collar:
        return 'White-collar'
    elif x in blue_collar:
        return 'Blue-collar'
    elif x in service:
        return 'Service'
    else:
        return 'unknown'

def simplify_relationship(value):
    if value in ['Husband', 'Wife']:
        return 'Spouse'
    elif value in ['Unmarried', 'Not-in-family']:
        return 'Single/Unmarried'
    elif value in ['Own-child', 'Other-relative']:
        return 'Dependent'
    else:
        return value

def simplify_race(x):
    if x == 'White':
        return 'White'
    elif x == 'Black':
        return 'Black'
    else:
        return 'other'

def capital_gain_category(x):
    return 'gain' if x > 0 else 'not gain'

def capital_loss_category(x):
    return 'loss' if x > 0 else 'not loss'


# Define file names
model_filename = 'income_best_model.pkl'
encoders_filename = 'encoders.pkl'
scaler_filename = 'scaler.pkl'
numerical_means_filename = 'numerical_means.pkl'
categorical_modes_filename = 'categorical_modes.pkl'
gain_map_filename = 'gain_map.pkl'
loss_map_filename = 'loss_map.pkl'
sex_map_filename = 'sex_map.pkl'

# Load model
with open(model_filename, 'rb') as file:
    best_model = pickle.load(file)

# Load encoders
with open(encoders_filename, 'rb') as file:
    encoders = pickle.load(file)

# Load scaler
with open(scaler_filename, 'rb') as file:
    scaler = pickle.load(file)

# Load numerical means
with open(numerical_means_filename, 'rb') as file:
    numerical_means = pickle.load(file)

# Load categorical modes
with open(categorical_modes_filename, 'rb') as file:
    categorical_modes = pickle.load(file)

# Load mapping dictionaries

with open(gain_map_filename, 'rb') as file:
    gain_map = pickle.load(file)

with open(loss_map_filename, 'rb') as file:
    loss_map = pickle.load(file)

with open(sex_map_filename, 'rb') as file:
    sex_map = pickle.load(file)

features_columns =  ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']
dropped_columns =  ['fnlwgt', 'education']
num_cols =  ['age', 'education.num', 'hours.per.week']
cat_cols =  ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'native.country']
nominal = ['workclass', 'native.country', 'occupation', 'relationship', 'race', 'marital.status']
binary = ['sex', 'capital.gain', 'capital.loss']

input_data = "90, Private, 154374, HS-grad, 16, Married-civ-spouse, Machine-op-inspct, Husband, White, Male, 0, 0, 40, United-States"

values = input_data.split(',')

# Function to check if a value is numeric
def enclose_value(value):
    # Handle missing values
    if value == '' or value is None:
        return np.nan
    # Check if the value is numeric (int or float)
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    # Return non-numeric value with single quotes
    except ValueError:
        return f'{value}'

processed_values = [enclose_value(value) for value in values]

if len(processed_values) != len(features_columns):
    raise ValueError("Input data does not match the expected number of features.")

# Convert the processed values into a DataFrame
input_data = pd.DataFrame([processed_values], columns=features_columns)

# remove dropped columns
input_data = input_data.drop(columns=dropped_columns)

# Deployment stage

# age
input_data['age'] = input_data['age'].clip(upper=75)
print(input_data['age'])

# hours.per.week
input_data['hours.per.week'] = input_data['hours.per.week'].clip(lower=10, upper=70)
print(input_data['hours.per.week'])

# for native country
user_country = input_data.loc[0, 'native.country']
category = simplify_native_country(user_country.strip())
input_data.loc[0, 'native.country'] = category

print("native country:", category)

# for workclass
user_workclass = input_data.loc[0, 'workclass']
category = simplify_workclass(user_workclass.strip())
input_data.loc[0, 'workclass'] = category

print("workclass:", category)

# for occupation
user_occupation = input_data.loc[0, 'occupation']
category = simplify_occupation(user_occupation.strip())
input_data.loc[0, 'occupation'] = category

print("occupation:", category)

# for occupation
user_relation = input_data.loc[0, 'relationship']
category = simplify_relationship(user_relation.strip())
input_data.loc[0, 'relationship'] = category

print("relationship:", category)

# for race
user_race = input_data.loc[0, 'race']
category = simplify_race(user_race.strip())
input_data.loc[0, 'race'] = category

print("race:", category)

# for race
user_marital = input_data.loc[0, 'marital.status']
category = simplify_marital(user_marital.strip())
input_data.loc[0, 'marital.status'] = category

print("marital.status:", category)


# for sex feature
user_sex = input_data.loc[0, 'sex'].strip()
encoded_value = sex_map[user_sex]
input_data.loc[0, 'sex'] = encoded_value
print("User sex:", user_sex)
print("Encoded sex:", encoded_value)



# # for capital gain

user_gain = input_data.loc[0, 'capital.gain']
category = capital_gain_category(user_gain)
encoded_value = gain_map[category]
input_data.loc[0, 'capital.gain'] = encoded_value

print('user gain:', user_gain)
print("gain Category:", category)
print("gain Encoded:", encoded_value)


# # for capital loss

user_loss = input_data.loc[0, 'capital.loss']
category = capital_loss_category(user_loss)
encoded_value = loss_map[category]
input_data.loc[0, 'capital.loss'] = encoded_value

print('user gain:', user_loss)
print("gain Category:", category)
print("gain Encoded:", encoded_value)


# Handle Missing Values
input_data[num_cols] = input_data[num_cols].fillna(numerical_means)
input_data[cat_cols] = input_data[cat_cols].fillna(categorical_modes)

input_data[num_cols] = scaler.transform(input_data[num_cols])

# One-hot encode nominal columns
nominal_encoded_array = encoders['onehot'].transform(input_data[nominal])
nominal_encoded_df = pd.DataFrame(nominal_encoded_array,
                                  columns=encoders['onehot'].get_feature_names_out(nominal),
                                  index=input_data.index)

input_data = input_data.drop(columns=nominal)
input_data = pd.concat([input_data, nominal_encoded_df], axis=1)


# Make Predictions
try:
    print("Making Prediction...")
    prediction = best_model.predict(input_data)
    if prediction == 0.0:
        print("Predicted Income: <=50K")
    else:
        print("Predicted Income: >50k")
except Exception as e:
    print("Error during Prediction:")
    print(f"An error occurred during prediction: {e}")

