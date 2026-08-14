# DS605 Lab 2 - NumPy and Pandas

## Assignment Details

**Name:** Steffi George  
**Student ID:** 202618018  
**Course:** DS605  
**Lab:** Lab 2

## Dataset

The Titanic dataset (`train.csv`) was used for the Pandas data-wrangling tasks.

The dataset contains information about Titanic passengers, including their age, sex, passenger class, fare, family information, embarkation point, and survival status.

## Project Details

This lab covers:

- Vectorized programming using NumPy
- Arrays and array operations
- Statistics and indexing
- Matrix operations and linear algebra
- Normal distribution and histograms
- Loading and inspecting data using Pandas
- Filtering and querying data
- Grouping and aggregation
- Missing-value handling
- Outlier detection using the IQR method
- Feature creation
- Pivot tables
- Data visualization
- Correlation analysis

## Key Observations

1. Female passengers had a much higher chance of surviving than male passengers. About 74.20% of females survived, while only about 18.89% of males survived.

2. Passenger class affected the chances of survival. Passengers in 1st class generally had a higher survival rate than passengers in 2nd and 3rd class.

3. Pclass and Fare had a negative relationship because 1st class has a lower class number but passengers in 1st class usually paid higher fares.

4. SibSp, Parch, and FamilySize are related because FamilySize is calculated using SibSp and Parch along with the passenger themselves.

5. Age and Fare do not have a strong relationship. Passengers of different ages paid different fares.

6. Passengers who paid higher fares were often more likely to survive, although fare alone did not determine survival.

7. Age and Fare together could not clearly separate survivors from non-survivors.