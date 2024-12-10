# %% [markdown]
# Loading Data

# %%
import pandas as pd
data = pd.read_csv("nutrition.csv")

# Feature Extraction
data[['Latitude', 'Longitude']] = data['GeoLocation'].str.extract(r'\(([^,]+),\s*([^)]+)\)')
data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')
data[['StratificationCategory', 'StratificationValue']] = data[['StratificationCategory1', 'Stratification1']]
data[['Income_Lower', 'Income_Upper']] = data['Income'].str.extract(r'\$(\d{1,3}(?:,\d{3})*)\s*-\s*\$(\d{1,3}(?:,\d{3})*)')
data['Income_Lower'] = data['Income_Lower'].str.replace(',', '').astype(float, errors='ignore')
data['Income_Upper'] = data['Income_Upper'].str.replace(',', '').astype(float, errors='ignore')

# %% [markdown]
# Data Characteristics

# %%

print("The shape os the dataset is : \n",data.shape)
print("Datatype of each attribute: \n",data.dtypes)
print("Sample attributes of dataset: \n",data.sample(7))

# %% [markdown]
# Data Collection and Manipulation

# %%
#Missing Values
def missing_values():
    missing_values = data.isnull().sum()
    print("missing values of each attribute : \n",missing_values)
    missing_percentage = data.isnull().mean() * 100
    print(missing_percentage)
missing_values()

# %%
#Dupliate Rows
def duplicate():
    duplicates = data.duplicated().sum()
    print("Duplicates attributes in dataset are ",duplicates)

duplicate()

# %%

print(data["Total"].unique())
print(data["Data_Value_Footnote_Symbol"].unique())
print(data["Data_Value_Footnote"].unique())
print(data["Gender"].unique())
print(len(data["Data_Value"].unique()))
print(len(data["Data_Value_Alt"].unique()))
#Dropping Attributes
columns_to_drop = ["Total", "Data_Value_Footnote_Symbol", "Data_Value_Footnote","Gender","Data_Value_Alt","Income_Lower","Income_Upper"]
data = data.drop(columns=columns_to_drop)

# %%
def missing_summary():
    missing_summary = pd.DataFrame({
        "Data Type": data.dtypes,
        "Missing Values Count": data.isnull().sum(),
        "Missing Values Percentage": data.isnull().mean() * 100
        })  
    print(missing_summary)
missing_summary()


# %%
#Handling missing values of categorical attributres 
data["Income"] = data["Income"].fillna("Missing Income")
data["Education"] = data["Education"].fillna("Missing Education")
data["Race/Ethnicity"] = data["Race/Ethnicity"].fillna("Other")
data["Age(years)"] = data["Age(years)"].fillna("Missing age group")
data=data.dropna(subset=["GeoLocation"])
#Handling missing values of numerical attributes
from sklearn.impute import SimpleImputer
numerical_columns = data.select_dtypes(include=["float64","int64"]).columns
imputer = SimpleImputer(strategy="median")
data[numerical_columns]=imputer.fit_transform(data[numerical_columns])
missing_summary()


# %% [markdown]
# 

# %%
# Outliers in Numerical attributes
num_columns = data.select_dtypes(include=['float64','int64']).columns

def outliers():
    Q1 = data[num_columns].quantile(0.25)
    Q3 = data[num_columns].quantile(0.75)
    IQR = Q3 -Q1
    outliers = ((data[num_columns] < (Q1 - 1.5 * IQR )) | (data[num_columns] > (Q3 + 1.5 * IQR))).sum()
    outliers_percentage = (outliers / len(data)) * 100
    print("Outliers in numerical attributes \n",outliers)
    print("outliers percentage in numerical attributes\n",outliers_percentage)
outliers()


# %%
# Visuallizing  outliers of numerical attributes before handling 
import matplotlib.pyplot as plt
import seaborn as sns
def visualize_outliers():
    plt.figure(figsize=(10,10))
    for i, col in enumerate(num_columns, 1):
        plt.subplot(5,2,i)
        sns.boxplot(data[col])
        plt.title(f"Box plot for {col}")
        plt.tight_layout()


    plt.show

    plt.figure(figsize=(10,10))
    for i, col in enumerate(num_columns, 1):
        plt.subplot(5, 2, i)
        plt.scatter(data.index, data[col], color='blue', alpha=0.5)
        plt.title(f"Scatter plot for {col}")
        plt.tight_layout()
visualize_outliers()


# %%
pd.options.display.max_columns = None
data.head()

# %%

print (data["Race/Ethnicity"].value_counts(),"\n")
print (data["Class"].value_counts(),"\n")
print (data["Data_Value"].value_counts(),"\n")



# %%
data = data.drop(columns = ['Topic'])
data.columns

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()

categorical_columns = [col for col in categorical_columns if col != "Class"]
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

print(data.head())

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_numeric = data.drop(columns=['Class'])
corr_matrix = data_numeric.corr()

plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Correlation Heatmap (Excluding Class)')
plt.show()


# %%
import pandas as pd

physical_activity = data[data["Class"] == "Physical Activity"]
obesity_weight_status = data[data["Class"] == "Obesity / Weight Status"]
fruits_vegetables = data[data["Class"] == "Fruits and Vegetables"]

print("Physical Activity DataFrame:")
print(physical_activity.head())

print("Obesity / Weight Status DataFrame:")
print(obesity_weight_status.head())

print("Fruits and Vegetables DataFrame:")
print(fruits_vegetables.head())


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

y = physical_activity['Data_Value']
X = physical_activity[["High_Confidence_Limit","Latitude","Longitude","StratificationCategory1"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R-squared: {r2_score(y_test, y_pred):.2f}")


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

y = obesity_weight_status['Data_Value']
X = obesity_weight_status[["High_Confidence_Limit","Latitude","Longitude","StratificationCategory1"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R-squared: {r2_score(y_test, y_pred):.2f}")


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

y = fruits_vegetables['Data_Value']
X = fruits_vegetables[["High_Confidence_Limit","Latitude","Longitude","StratificationCategory1"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R-squared: {r2_score(y_test, y_pred):.2f}")


# %%
import matplotlib.pyplot as plt

attributes = ["High_Confidence_Limit", "Latitude", "Longitude", "StratificationCategory1"]
plt.figure(figsize=(12, 10))

for i, col in enumerate(attributes, 1):
    plt.subplot(2, 2, i)
    plt.hist(physical_activity[col], bins=20, color="skyblue", edgecolor="black")
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# %%
import seaborn as sns

plt.figure(figsize=(12, 10))

for i, col in enumerate(attributes, 1):
    plt.subplot(2, 2, i)
    sns.kdeplot(physical_activity[col], fill=True, color="green", alpha=0.7)
    plt.title(f"KDE Plot of {col}")
    plt.xlabel(col)
    plt.ylabel("Density")

plt.tight_layout()
plt.show()


