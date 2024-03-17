import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# import data
df = pd.read_csv('Dataset test technique FRINGUANT - Data Analyst.csv')

# Discovery of Data
print(df.head())
print(df.info())
print(df.describe())

# Data Cleaning

## Check Duplicates
print(df.duplicated().sum())  # 0 duplicated values

## Check Null Values
print(df.isnull().sum())
### 2 null values in Satisfaction Level, let's check
print(df[df[['Satisfaction Level']].isnull().T.any()][['Customer ID', 'Average Rating', 'Satisfaction Level']])
### There are values in average rating, but not in satisfaction level.
### We need to get the satisfaction level by their rate.

## Check mean and median of average rating.
print(df.groupby('Satisfaction Level')['Average Rating'].mean())
print(df.groupby('Satisfaction Level')['Average Rating'].median())
### Their average rating are 3.1 & 3.4, which are probably matching the satisfaction level Neutral.

df['Satisfaction Level'].fillna('Neutral', inplace=True)
print(df.info())  # There's no null values in dataset.

# Data Analysis
# 1. Demographic Situation
CustomerByGender = df['Gender'].value_counts().reset_index()
print(CustomerByGender)

fig1 = px.pie(
    CustomerByGender,
    names='index',
    values='Gender',
    title='Gender Distribution',
    labels={'index': 'Gender', 'Gender': 'Count'}
)
# fig1.show()

CustomerByAge = df['Age'].value_counts().reset_index()
print(CustomerByAge)
fig1_2 = px.histogram(
    CustomerByAge,
    x='index',
    y='Age',
    nbins=20,
    title='Age Distribution',
    labels={'index': 'Age', 'Age': 'Count'}
)
fig1_2.update_layout(bargap=0.2)
# fig1_2.show()

# 2. Geographic Distribution
CustomerByCity = df['City'].value_counts().reset_index()
print(CustomerByCity)

fig2 = px.bar(
    CustomerByCity,
    x='index',
    y='City',
    labels={'index': 'City', 'City': 'Customer Number'},
    title='Customer Distribution by City',
    color='index'
)
# fig2.show()

# 3. Satisfaction Level Analysis
Satisfaction = df['Satisfaction Level'].value_counts().reset_index()
print(Satisfaction)

fig3 = px.bar(
    Satisfaction,
    x='index',
    y='Satisfaction Level',
    color='index',
    labels={'index': 'Satisfaction Level', 'Satisfaction Level': 'Count'},
    title='Satisfaction Level Distribution'
)
# fig3.show()

# 4. Membership
MembershipByCustomer = df['Membership Type'].value_counts().reset_index()
print(MembershipByCustomer)
fig4 = px.bar(
    MembershipByCustomer,
    x='index',
    y='Membership Type',
    labels={'index': 'Membership', 'Membership Type': 'Count'},
    title='Membership Distribution'
)
# fig4.show()

# 5. Correlations
Correlation = df.corr()
print(Correlation)
fig5 = px.imshow(
    Correlation
)
# fig5.show()

# Metric & KPIs
# 1. Average Spend
AverageSpend = df['Total Spend'].sum() / df['Customer ID'].count()
print(AverageSpend)  # 845.38

# 2. Average Quantity
AverageQuantity = df['Items Purchased'].sum() / df['Customer ID'].count()
print(AverageQuantity)  # 12.6

# 3. Conversion Rate
ConversionRate = df[df['Items Purchased'] > 0]['Customer ID'].nunique() / df['Customer ID'].count() * 100
print(ConversionRate)

# 4. Retention Rate Last 30 days
Active = df[df['Days Since Last Purchase'] <= 30]['Customer ID'].nunique()
print(Active)  # 226 Active
RetentionRate = Active / df['Customer ID'].nunique() * 100
print(RetentionRate)  # 64.57%

# 5. Customer with Discount
DiscountedCustomer = df[df['Discount Applied'] == True]['Customer ID'].count()
print(DiscountedCustomer)  # 175

# 6. Median of Spend
MedianSpend = df['Total Spend'].median()
print(MedianSpend)  # 775.2

# Average Spending Situation by Membership & Gender
AvgByMember = df.groupby(['Gender', 'Membership Type'])['Total Spend'].mean().reset_index()
print(AvgByMember)
fig6 = sns.barplot(
    AvgByMember,
    x='Membership Type',
    y='Total Spend',
    hue='Gender'
)
plt.show()

# Top 3 Cities by Average Spending
AverageSpendByCity = df.groupby('City')['Total Spend'].mean().reset_index().sort_values('Total Spend', ascending=False)
AverageSpendByCity.columns = ['City', 'Average Spend']
print(AverageSpendByCity)
fig7 = px.bar(
    AverageSpendByCity,
    x='Average Spend',
    y='City',
    color='City',
    title='Average Spend by City'
)
# fig7.show()

# Average Rating Analysis : Mean, By Membership & Gender, By Age
AverageRating = df['Average Rating'].mean()
print(AverageRating)  # 4.01

AverageRatingByMembership = df.groupby(['Membership Type', 'Gender'])[
    'Average Rating'].mean().reset_index().sort_values('Average Rating', ascending=False)
print(AverageRatingByMembership)
fig8 = sns.barplot(
    AverageRatingByMembership,
    x='Membership Type',
    y='Average Rating',
    hue='Gender'
)
plt.show()

fig9 = px.scatter(
    df,
    x=df['Age'],
    y=df['Average Rating'],
    size=df['Total Spend'],
    title='Average Rating by Age & Total Spend'
)
fig9.show()

# Predictive Modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = df[['Age', 'Total Spend', 'Items Purchased', 'Days Since Last Purchase', 'Discount Applied']]
y = df['Average Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Model = LinearRegression()
Model.fit(X_train, y_train)
y_pred = Model.predict(X_test)
print(y_pred)

mse = mean_squared_error(y_test, y_pred)
print(mse)  # 0.0338