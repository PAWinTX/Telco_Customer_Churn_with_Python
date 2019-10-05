"""
Customer Churn Analysis of Watson Analytics telecommunications customer data
Software: Python 3.6.4
File: Telco_analysis_v1.1.py
Developer: Paul A. Wilson
Initial dev: 03/31/2018
Last revision: 10/05/2019
"""

#Begin Data import, initial exploration and cleaning #####################################

#Import libraries and modules
import pandas as pd

#Extract customer data (cd) from GitHub (originally hosted by IBM for Watson Analytics)
cd = pd.read_csv('https://raw.githubusercontent.com/PAWinTX/Telco_Customer_Churn_with_Python/master/local_WA_Fn-UseC_-Telco-Customer-Churn.csv')

#Optional - Write a copy of cd to disc in CSV format
cd.to_dense().to_csv('C:\Github\Telco\local_WA_Fn-UseC_-Telco-Customer-Churn.csv', index = False, sep=',', encoding='utf-8')

#Count of observations and columns
print(cd.shape)

#View first five rows
print(cd.head())

#Statistical Identity of variables
#DataFrame info: count of observations and variables, count of non-missing values, and datatypes of each variable
print(cd.info())

#Explore target variable Churn
#Count of all values within variable Churn
print(cd.Churn.value_counts(dropna=False))

#Mean of numerical variables across the two classes of Churn
cd.groupby('Churn').mean()

#Data cleaning

#Search for duplicate observations
#Five most frequent values of CustomerID
print(cd.customerID.value_counts(dropna=False).head())

#Search for any duplicate observations across all variables
cd_dup = cd.duplicated(keep='first') == bool('True')
print(cd[cd_dup])

#Delete customerID - not needed
cd.drop(columns=['customerID'], inplace=True)
#Verify
print(cd.columns)

#Verify absence of nulls
print(cd.isnull().sum())

#Rename variables with correct case
cd.rename(index=str, columns={'gender':'Gender', 'tenure': 'Tenure'}, inplace=True)
#Verify
print(cd.columns)

#Transform Text values to binary
#Transform Male to 1 and Female to 0
print(cd.Gender.value_counts(dropna=False))
cd['Gender'].replace('Male', '1', inplace=True)
cd['Gender'].replace('Female', '0', inplace=True)
print(cd.Gender.value_counts(dropna=False))
#Transform Partner Yes to 1 and No to 0
print(cd.Partner.value_counts(dropna=False))
cd['Partner'].replace('Yes', '1', inplace=True)
cd['Partner'].replace('No', '0', inplace=True)
print(cd.Partner.value_counts(dropna=False))
#Transform Dependents Yes to 1 and No to 0
print(cd.Dependents.value_counts(dropna=False))
cd['Dependents'].replace('Yes', '1', inplace=True)
cd['Dependents'].replace('No', '0', inplace=True)
print(cd.Dependents.value_counts(dropna=False))
#Transform PhoneService Yes to 1 and No to 0
print(cd.PhoneService.value_counts(dropna=False))
cd['PhoneService'].replace('Yes', '1', inplace=True)
cd['PhoneService'].replace('No', '0', inplace=True)
print(cd.PhoneService.value_counts(dropna=False))
#Transform PaperlessBilling Yes to 1 and No to 0
print(cd.PaperlessBilling.value_counts(dropna=False))
cd['PaperlessBilling'].replace('Yes', '1', inplace=True)
cd['PaperlessBilling'].replace('No', '0', inplace=True)
print(cd.PaperlessBilling.value_counts(dropna=False))
#Transform Churn Yes to 1 and No to 0
print(cd.Churn.value_counts(dropna=False))
cd['Churn'].replace('Yes', '1', inplace=True)
cd['Churn'].replace('No', '0', inplace=True)
print(cd.Churn.value_counts(dropna=False))

#Convert binary object variables to int64
for name in ['Gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']:
    cd[name] = cd[name].astype('int64')

#Convert remaining object variables to category
for name in ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']:
    cd[name] = cd[name].astype('category') 

#Explore most values in TotalCharges to find any non-numerical
print(cd.TotalCharges.value_counts(dropna=False).head(-1))

#Find rows with erroneous spaces in TotalCharges
TtlChg_space = cd['TotalCharges'] == ' '
print(cd[TtlChg_space][['Tenure', 'TotalCharges']])

#Convert spaces to 0
cd['TotalCharges'].replace(' ', '0', inplace=True)

#Convert TotalCharges variable from object to float64
cd['TotalCharges'] = cd['TotalCharges'].astype('float64')

#Search for most frequent values again
print(cd.TotalCharges.value_counts(dropna=False).head())    

#Verify conversions and note reduced memory usage
print(cd.info())

#Optional - Export a copy of cleaned and prepared data in Excel format
#Create Pandas Excel writer
writer = pd.ExcelWriter('C:\Github\Telco\cleaned_WA_Fn-UseC_-Telco-Customer-Churn.xlsx', engine='xlsxwriter')
#Convert the dataframe to an XlsxWriter Excel object
cd.to_excel(writer, sheet_name='CustData', index=False)
# Close the Pandas Excel writer and output the Excel file
writer.save()

#Optional - Write a copy of cd to disc in CSV format
cd.to_dense().to_csv('C:\Github\Telco\cleaned_WA_Fn-UseC_-Telco-Customer-Churn.csv', index = False, sep=',', encoding='utf-8')

#End Data import, initial exploration and cleaning ######################################

#Begin Exploratory Data analysis ########################################################

#Import libraries and modules
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

#Set plot and grid style.
sns.set(style="whitegrid")
#sns.set_style("whitegrid")

#Distribution of variables using univariate stats

#Summary statistics for numerical variables
print(cd.describe())

#Boxplot for Tenure
plt.boxplot(cd.Tenure)
plt.ylabel('Months')
plt.title('Boxplot of Months Tenure')
plt.show()
#Histogram for Tenure
plt.hist(cd.Tenure)
plt.xlabel('Months')
plt.ylabel('Customers')
plt.title('Histogram of Months Tenure')
plt.show()

#Recalculate Tenure from TotalCharges and MonthlyCharges
cd['TenureCalc'] = round(cd.TotalCharges / cd.MonthlyCharges)
#Summary stats for Tenure and TenureCalc
print(cd[['Tenure', 'TenureCalc']].describe())
#Boxplot for Tenure
plt.subplot(2,2,1)
plt.boxplot(cd.Tenure)
plt.ylabel('Months')
plt.title('Boxplot of Months Tenure')
#Boxplot for TenureCalc
plt.subplot(2,2,2)
plt.boxplot(cd.TenureCalc)
plt.ylabel('Months')
plt.title('Boxplot of Months TenureCalc')
#Histogram for Tenure
plt.subplot(2,2,3)
plt.hist(cd.Tenure)
plt.xlabel('Months')
plt.ylabel('Customers')
plt.title('Histogram of Months Tenure')
#Histogram for TenureCalc
plt.subplot(2,2,4)
plt.hist(cd.TenureCalc)
plt.xlabel('Months')
plt.ylabel('Customers')
plt.title('Histogram of Months TenureCalc')
plt.tight_layout()
plt.show()

#Boxplot for TotalCharges
plt.boxplot(cd.TotalCharges)
plt.ylabel('$ Total Charges')
plt.title('Boxplot of Total Charges')
plt.show()
#Histogram for TotalCharges
plt.hist(cd.TotalCharges)
plt.xlabel('$ Total Charges')
plt.ylabel('Customers')
plt.title('Histogram of Total Charges')
plt.show()

#Boxplot for MonthlyCharges
plt.boxplot(cd.MonthlyCharges)
plt.ylabel('$ Monthly Charges')
plt.title('Boxplot of Monthly Charges')
plt.show()
#Histogram for MonthlyCharges
plt.hist(cd.MonthlyCharges)
plt.xlabel('$ Monthly Charges')
plt.ylabel('Customers')
plt.title('Histogram of Monthly Charges')
plt.show()

#Remaining quantitative and qualitative categorical variables

#Customer demographics
#Gender
fig, ax = plt.subplots()
male = mpatches.Patch(color='C0', label='Male')
female = mpatches.Patch(color='C1', label='Female')
plt.legend(handles=[female,male])
print(cd.Gender.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Gender'))
plt.xlabel('Customers')
plt.ylabel('Sex')
plt.legend(handles=[male,female], bbox_to_anchor=(0., 1.08, 1., .102), ncol=2, borderaxespad=0.)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.3, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Senior citizen
fig, ax = plt.subplots()
no = mpatches.Patch(color='C0', label='Not a Senior')
yes = mpatches.Patch(color='C1', label='Senior')
plt.legend(handles=[no,yes])
print(cd.SeniorCitizen.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Senior Citizen'))
plt.xlabel('Customers')
plt.ylabel('Senior')
plt.legend(handles=[no,yes], bbox_to_anchor=(0., 1.08, 1., .102), ncol=2, borderaxespad=0.)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.3, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Partner
fig, ax = plt.subplots()
no = mpatches.Patch(color='C0', label='No Partner')
yes = mpatches.Patch(color='C1', label='Partner')
plt.legend(handles=[no,yes])
print(cd.Partner.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Partner'))
plt.xlabel('Customers')
plt.ylabel('Partner')
plt.legend(handles=[no, yes], bbox_to_anchor=(0., 1.08, 1., .102), ncol=2, borderaxespad=0.)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.3, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Dependents
fig, ax = plt.subplots()
no = mpatches.Patch(color='C0', label='No Dependents')
yes = mpatches.Patch(color='C1', label='Dependents')
plt.legend(handles=[no,yes])
print(cd.Dependents.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Dependents'))
plt.xlabel('Customers')
plt.ylabel('Dependents')
plt.legend(handles=[no,yes], bbox_to_anchor=(0., 1.08, 1., .102), ncol=2, borderaxespad=0.)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.3, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Phone service
fig, ax = plt.subplots()
no = mpatches.Patch(color='C1', label='No Phone Service')
yes = mpatches.Patch(color='C0', label='Phone Service')
plt.legend(handles=[no,yes])
print(cd.PhoneService.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Phone Service'))
plt.xlabel('Customers')
plt.ylabel('Phone Service')
plt.legend(handles=[yes,no], bbox_to_anchor=(0., 1.08, 1., .102), ncol=2, borderaxespad=0.)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.3, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Customer account data variables
#Multiple lines
fig, ax = plt.subplots()
print(cd.MultipleLines.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Multiple Lines'))
plt.xlabel('Customers')
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.35, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Online service variables
#Internet service
fig, ax = plt.subplots()
print(cd.InternetService.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Internet Service'))
plt.xlabel('Customers')
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.35, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Online security
fig, ax = plt.subplots()
print(cd.OnlineSecurity.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Online Security'))
plt.xlabel('Customers')
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.35, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Online backup
fig, ax = plt.subplots()
print(cd.OnlineBackup.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Online Backup'))
plt.xlabel('Customers')
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.35, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Device protection
fig, ax = plt.subplots()
print(cd.DeviceProtection.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Device Protection'))
plt.xlabel('Customers')
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.35, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Tech support
fig, ax = plt.subplots()
print(cd.TechSupport.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Tech Support'))
plt.xlabel('Customers')
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.35, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Streaming TV
fig, ax = plt.subplots()
print(cd.StreamingTV.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Streaming TV'))
plt.xlabel('Customers')
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.35, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Streaming movies
fig, ax = plt.subplots()
print(cd.StreamingMovies.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Streaming Movies'))
plt.xlabel('Customers')
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.35, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Contract, Billing, and Payment variables
#Contract
fig, ax = plt.subplots()
print(cd.Contract.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Contract'))
plt.xlabel('Customers')
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.35, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Paperless billing
fig, ax = plt.subplots()
no = mpatches.Patch(color='C1', label='No Paperless Billing')
yes = mpatches.Patch(color='C0', label='Paperless Billing')
plt.legend(handles=[no,yes])
print(cd.PaperlessBilling.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Paperless Billing'))
plt.xlabel('Customers')
plt.ylabel('Paperless Billing')
plt.legend(handles=[no,yes], bbox_to_anchor=(0., 1.08, 1., .102), ncol=2, borderaxespad=0.)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.17, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
plt.show()

#Payment method
fig, ax = plt.subplots()
print(cd.PaymentMethod.value_counts(dropna=False)[:10].plot(kind='barh', figsize=(10, 2), grid=True, zorder=2, title='Payment Method'))
plt.xlabel('Customers')
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.35, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Dependent variable Churn
#Churn
fig, ax = plt.subplots()
no = mpatches.Patch(color='C0', label='No Churn')
yes = mpatches.Patch(color='C1', label='Churn')
plt.legend(handles=[no,yes])
print(cd.Churn.value_counts(dropna=False).plot(kind='barh', figsize=(11, 2), grid=True, zorder=2, title='Churn'))
plt.xlabel('Customers')
plt.ylabel('Churn')
plt.legend(handles=[no,yes], bbox_to_anchor=(0., 1.08, 1., .102), ncol=2, borderaxespad=0.)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.17, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#Bivariate analysis

#Figure size in inches
plt.rcParams['figure.figsize'] = 10,7

#Churn by continuous variables
#By TenureCalc
sns.violinplot(x='Churn',y='TenureCalc', data=cd, scale='count')
plt.xlabel('Churn: 0=No 1=Yes')
plt.ylabel('TenureCalc')
plt.title('Customer Churn by TenureCalc')
plt.show()
#By MonthlyCharges
sns.violinplot(x='Churn',y='MonthlyCharges', data=cd, scale='count')
plt.xlabel('Churn: 0=No 1=Yes')
plt.ylabel('Monthly Charges')
plt.title('Customer Churn by Monthly Charges')
plt.show()
#By TotalCharges
sns.violinplot(x='Churn',y='TotalCharges', data=cd, scale='count')
plt.xlabel('Churn: 0=No 1=Yes')
plt.ylabel('Total Charges')
plt.title('Customer Churn by Total Charges')
plt.show()

#Churn by demographic variables
#By Gender
fig, ax = plt.subplots(figsize=(10.5,3))
male = mpatches.Patch(color='C0', label='Male')
female = mpatches.Patch(color='C1', label='Female')
plt.legend(handles=[female,male])
print(sns.countplot(y='Churn', hue='Gender', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0-No 1-Yes')
plt.title('Customer Churn by Gender')
plt.legend(handles=[female,male], loc=1)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.17, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#By SeniorCitizen
fig, ax = plt.subplots(figsize=(10.5,3))
no = mpatches.Patch(color='C0', label='Not a Senior')
yes = mpatches.Patch(color='C1', label='Senior')
plt.legend(handles=[no,yes])
print(sns.countplot(y='Churn', hue='SeniorCitizen', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0-No 1-Yes')
plt.title('Customer Churn by Senior Citizen')
plt.legend(handles=[yes,no], loc=1)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.17, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#By Partner
fig, ax = plt.subplots(figsize=(10.5,3))
no = mpatches.Patch(color='C0', label='No Partner')
yes = mpatches.Patch(color='C1', label='Partner')
plt.legend(handles=[no,yes])
print(sns.countplot(y='Churn', hue='Partner', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0-No 1-Yes')
plt.title('Customer Churn by Partner')
plt.legend(handles=[yes,no], loc=1)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.17, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#By Dependents
fig, ax = plt.subplots(figsize=(10.5,3))
no = mpatches.Patch(color='C0', label='No Dependents')
yes = mpatches.Patch(color='C1', label='Dependents')
plt.legend(handles=[no,yes])
print(sns.countplot(y='Churn', hue='Dependents', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0-No 1-Yes')
plt.title('Customer Churn by Dependents')
plt.legend(handles=[yes,no], loc=1)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.17, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#By PhoneService
fig, ax = plt.subplots(figsize=(10.5,3))
no = mpatches.Patch(color='C0', label='No Phone Service')
yes = mpatches.Patch(color='C1', label='Phone Service')
plt.legend(handles=[no,yes])
print(sns.countplot(y='Churn', hue='PhoneService', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0-No 1-Yes')
plt.title('Customer Churn by Phone Service')
plt.legend(handles=[yes,no], loc=1)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.17, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#By MultipleLines
fig, ax = plt.subplots(figsize=(10.5,3))
print(sns.countplot(y='Churn', hue='MultipleLines', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0=No 1=Yes')
plt.title('Customer Churn by Multiple Lines')
plt.legend(loc=1)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.09, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#By InternetService
fig, ax = plt.subplots(figsize=(10.5,3))
print(sns.countplot(y='Churn', hue='InternetService', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0=No 1=Yes')
plt.title('Customer Churn by Internet Service')
plt.legend(loc=1)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.09, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#By OnlineSecurity
fig, ax = plt.subplots(figsize=(10.5,3))
print(sns.countplot(y='Churn', hue='OnlineSecurity', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0=No 1=Yes')
plt.title('Customer Churn by Online Security')
plt.legend(loc=1)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.09, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#By TechSupport
fig, ax = plt.subplots(figsize=(10.5,3))
print(sns.countplot(y='Churn', hue='TechSupport', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0=No 1=Yes')
plt.title('Customer Churn by Tech Support')
plt.legend(loc=1)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.09, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#By OnlineBackup
fig, ax = plt.subplots(figsize=(10.5,3))
print(sns.countplot(y='Churn', hue='OnlineBackup', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0=No 1=Yes')
plt.title('Customer Churn by Online Backup')
plt.legend(loc=1)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.09, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#By DeviceProtection
fig, ax = plt.subplots(figsize=(10.5,3))
print(sns.countplot(y='Churn', hue='DeviceProtection', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0=No 1=Yes')
plt.title('Customer Churn by Device Protection')
plt.legend(loc=1)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.09, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#By StreamingTV
fig, ax = plt.subplots(figsize=(10.5,3))
print(sns.countplot(y='Churn', hue='StreamingTV', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0=No 1=Yes')
plt.title('Customer Churn by Streaming TV')
plt.legend(loc=1)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.09, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#By StreamingMovies
fig, ax = plt.subplots(figsize=(10.5,3))
print(sns.countplot(y='Churn', hue='StreamingMovies', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0=No 1=Yes')
plt.title('Customer Churn by Streaming Movies')
plt.legend(loc=1)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.09, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#By Contract
fig, ax = plt.subplots(figsize=(10.5,3))
print(sns.countplot(y='Churn', hue='Contract', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0=No 1=Yes')
plt.title('Customer Churn by Contract')
plt.legend(loc=1)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.09, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#By PaperlessBilling
fig, ax = plt.subplots(figsize=(10.5,3))
no = mpatches.Patch(color='C0', label='No Paperless Billing')
yes = mpatches.Patch(color='C1', label='Paperless Billing')
plt.legend(handles=[no,yes])
print(sns.countplot(y='Churn', hue='PaperlessBilling', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0=No 1=Yes')
plt.title('Customer Churn by Paperless Billing')
plt.legend(handles=[yes,no], loc=1)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.17, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#By PaymentMethod
fig, ax = plt.subplots(figsize=(10.5,3))
print(sns.countplot(y='Churn', hue='PaymentMethod', data=cd, ax=ax))
plt.xlabel('Customers')
plt.ylabel('Churn: 0=No 1=Yes')
plt.title('Customer Churn by Payment Method')
plt.legend(bbox_to_anchor=(0., 1.3, 1., .102), ncol=1, borderaxespad=0.)
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.06, \
            str(round((i.get_width()), 2)), fontsize=11)
plt.margins(0.09)
ax.invert_yaxis()
plt.show()

#End Exploratory Data analysis ##########################################################


#Begin Predictive Data Analysis #########################################################

#Import libraries and modules
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
#cross_validation deprecated in version 0.18 in favor of the model_selection module
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

#Prepare data for tool use
#Delete Tenure from DataFrame cd - not needed
cd.drop(columns=['Tenure'], inplace=True)
#Create copy of cd 
cd_copy = cd.copy(deep=True)
#Delete dependent variable Churn from DataFrame cd_copy 
cd_copy.drop(columns=['Churn'], inplace=True)
#Create DataFrame cd_n with categorical variables converted to numerical
cd_n = pd.get_dummies(cd_copy)
#Create numpy array for data
cd_n_data = np.array(cd_n.values)
#Create numpy array for feature names
cd_n_feat = np.array(cd_n.keys())
#Create numpy array for dependent variable values
depVar_val = cd['Churn'].values

#PCA analytic method to find the optimal number of variables
#Instantiate a scaler
scaler = StandardScaler()
#Instantiate a PCA
pca = PCA()
#Create pipeline from scaler and pca
pipeline = make_pipeline(scaler, pca)
#Fit the pipeline to the data array
pipeline.fit(cd_n_data)
#Print the values of explained variances
print(np.round_(pca.explained_variance_,decimals=2))

#Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('Principal Component')
plt.ylabel('Variance')
plt.xticks(features)
plt.rcParams['figure.figsize'] = 10,7
plt.title('Principal Components by Variance')
plt.show()

#Plot the cumulative explained variance ratio
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance')
plt.rcParams['figure.figsize'] = 10,7
plt.title('Cumulative Ratio of Principal Components')
plt.show()

#Predictive methods to refine data and predict 
#Lasso 
#Instantiate a Lasso and set alpha (experiment with variations to find the most illustrative value in the next plot)
#lasso = Lasso(alpha=0.022)
lasso = Lasso(alpha=0.023)
#lasso = Lasso(alpha=0.024)

#Instantiate a Lasso coefficient to extract the coefficients
lasso_coef = lasso.fit(cd_n_data, depVar_val).coef_
#Plot the coefficients as a function of the feature names
plt.rcParams['figure.figsize'] = 11,7
plt.plot(range(len(cd_n_feat)), lasso_coef)
plt.xticks(range(len(cd_n_feat)), cd_n_feat, rotation=90)
plt.xlabel('Independent Variables')
plt.ylabel('Coefficients')
plt.title('Lasso Regression for Independent Variable Selection')
plt.show()

#Create a reduced variable DataFrame cd_red
cd_red = pd.DataFrame(cd_n_data, columns=cd_n_feat)
print(cd_red.columns)
#Drop unused columns to leave only five chosen variables
cd_red.drop(columns=['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
       'PaperlessBilling', 'TotalCharges', 'MultipleLines_No',
       'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_DSL',
       'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No', 
       'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No',
       'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No', 
       'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
       'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No',
       'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No', 
       'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year',
       'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Mailed check'], 
    inplace=True)

#Examine cd_red
cd_red.shape
cd_red.info()
cd_red.head()

#Create X and y data for logistic regression model
X = np.array(cd_red.values)
y = depVar_val

#Fitting the model
#Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Instantiate a logistics regressor
logreg = LogisticRegression()
#Fit to training set
logreg.fit(X_train, y_train)
#Predicting the test set results and calculating the accuracy
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
###Accuracy of logistic regression classifier on test set: 0.79 ###

#Cross validation attempts to avoid overfitting while still producing a prediction for each observation dataset
#Use 10-fold Cross-Validation to train the Logistic Regression model
#Instantiate a kfold cross validator
kfold = model_selection.KFold(n_splits=10, random_state=7)
#Instantiate a logistics regressor
modelCV = LogisticRegression()
#Score the cross validation
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
###10-fold cross validation average accuracy: 0.797 ###
#The average accuracy of 0.797 is very close to the Logistic Regression model accuracy of 0.79
#The model generalizes well

#Confusion Matrix for Churn = 1
print(confusion_matrix(y_test, y_pred, labels=[1,0]))
#Output:
#[[ 293  281 ]
# [ 159 1380 ]]

#[[ tn  fp ]
# [ fn tp ]]

#Compute precision, recall, F-measure and support
#Classification report
print(classification_report(y_test, y_pred, target_names=['1-Churn', '0-No Churn']))

#Calculate Receiver Operator Characteristic (ROC) curve
y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
logit_roc_auc = roc_auc_score(y_test, y_pred_prob)
#plot ROC
plt.figure()
plt.figure(figsize=(10,10))
plt.plot([0,1], [0,1], 'r--')
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.legend(loc="lower right")
plt.show()
#Logistic Regression AOC = 0.85

#Calculate odds
#print(logreg.intercept_)
print(np.exp(logreg.coef_))

#copy cd_red to new DataFrame
cd_final = cd_red

#append dependent variable Churn to DataFrame cd_final
cd_final['Churn'] = depVar_val

#Convert whole number float64 variables to int64
for name in ['TenureCalc', 'TechSupport_No', 'Contract_Month-to-month', 'PaymentMethod_Electronic check']:
    cd_final[name] = cd_final[name].astype('int64')

#Explore cd_final
cd_final.info()
cd_final.head()

#Five sample observations with characteristics of probable churn
chrn = cd_final['Churn'] == 0
mChg = cd_final['MonthlyCharges'] > 35
tnClc = cd_final['TenureCalc'] <= 12
tchSpt = cd_final['TechSupport_No'] == 1
cMtM = cd_final['Contract_Month-to-month'] == 1
pMtdEc = cd_final['PaymentMethod_Electronic check'] == 1
condition_all = chrn & mChg & tnClc & tchSpt & cMtM & pMtdEc
probChrn = cd_red[condition_all]
print(probChrn.head())

#End Predictive Data Analysis ###########################################################
