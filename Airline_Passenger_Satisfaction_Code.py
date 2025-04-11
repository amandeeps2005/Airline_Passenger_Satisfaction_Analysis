import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Airline_Passenger_Satisfaction_Dataset.csv")
df1 = df.copy()

# Fill missing values in service-related columns
service_cols = [
    'Departure and Arrival Time Convenience', 'Ease of Online Booking',
    'Check-in Service', 'Online Boarding', 'Gate Location', 'On-board Service',
    'Seat Comfort', 'Leg Room Service', 'Cleanliness', 'Food and Drink',
    'In-flight Service', 'In-flight Wifi Service', 'In-flight Entertainment',
    'Baggage Handling'
]
df1[service_cols] = df1[service_cols].apply(lambda x: x.fillna(x.mean()))

# Feature Engineering
df1['Satisfaction Binary'] = df1['Satisfaction'].apply(lambda x: 1 if x == 'Satisfied' else 0)
df1['Total Service Score'] = df1[service_cols].sum(axis=1)

# Age Group Segmentation
bins = [0, 19, 35, 55, 120]
labels = ['Teen', 'Adult', 'Middle-aged', 'Senior']
df1['Age Group'] = pd.cut(df1['Age'], bins=bins, labels=labels)

# ----------------- 4.1: Impact of Flight Distance -----------------
distance_bins = [0, 1000, 2000, df1['Flight Distance'].max()]
distance_labels = ['Short-haul', 'Medium-haul', 'Long-haul']
df1['Distance Group'] = pd.cut(df1['Flight Distance'], bins=distance_bins, labels=distance_labels)
distance_group_avg = df1.groupby('Distance Group', observed=False)[['Satisfaction Binary', 'Total Service Score']].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(data=df1, x='Distance Group', y='Satisfaction Binary', errorbar=None)
plt.title('Satisfaction Rate by Distance Group')
plt.xlabel('Flight Distance Group')
plt.ylabel('Satisfaction Rate')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.lineplot(data=distance_group_avg, x='Distance Group', y='Total Service Score', marker='o', linewidth=2.5)
plt.title('Total Service Score by Flight Distance')
plt.xlabel('Flight Distance Group')
plt.ylabel('Total Service Score')
plt.tight_layout()
plt.show()

# ----------------- 4.2: Satisfaction Rate by Class -----------------
plt.figure(figsize=(8, 5))
sns.barplot(data=df1, x='Class', y='Satisfaction Binary', errorbar=None)
plt.title('Satisfaction Rate by Class')
plt.xlabel('Class')
plt.ylabel('Satisfaction Rate')
plt.tight_layout()
plt.show()

# ----------------- 4.3: Lowest Rated In-Flight Services -----------------
avg_service_ratings = df1[service_cols].mean().sort_values()
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_service_ratings.values, y=avg_service_ratings.index, hue=avg_service_ratings.index,
            palette="Reds_r", legend=False)
plt.title('Lowest Rated In-Flight Services')
plt.xlabel('Average Rating')
plt.ylabel('Service')
plt.tight_layout()
plt.show()

# ----------------- 4.4: Service Ratings by Age Group -----------------
age_service_avg = df1.groupby('Age Group', observed=False)[service_cols].mean()

plt.figure(figsize=(14, 6))
sns.heatmap(age_service_avg, annot=True, cmap='YlGnBu', fmt=".1f", linewidths=0.5)
plt.title('Average Service Ratings by Age Group')
plt.ylabel('Age Group')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Line Plot for Selected Services
selected_services = ['Seat Comfort', 'Food and Drink', 'In-flight Entertainment']
plt.figure(figsize=(10, 5))
for service in selected_services:
    sns.lineplot(data=age_service_avg, x=age_service_avg.index, y=service, marker='o', label=service)

plt.title('Selected Service Ratings Across Age Groups')
plt.ylabel('Average Rating')
plt.xlabel('Age Group')
plt.legend()
plt.tight_layout()
plt.show()

# ----------------- 4.5: Delays vs Satisfaction -----------------
delay_df = df1[['Departure Delay', 'Arrival Delay', 'Satisfaction Binary']]
plt.figure(figsize=(6, 4))
sns.heatmap(delay_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation: Delays vs Satisfaction')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(data=df1, x='Departure Delay', y='Satisfaction Binary', ax=axs[0])
axs[0].set_title('Departure Delay vs Satisfaction')
axs[0].set_xlabel('Departure Delay')
axs[0].set_ylabel('Satisfaction')

sns.scatterplot(data=df1, x='Arrival Delay', y='Satisfaction Binary', ax=axs[1])
axs[1].set_title('Arrival Delay vs Satisfaction')
axs[1].set_xlabel('Arrival Delay')
axs[1].set_ylabel('Satisfaction')

plt.tight_layout()
plt.show()
