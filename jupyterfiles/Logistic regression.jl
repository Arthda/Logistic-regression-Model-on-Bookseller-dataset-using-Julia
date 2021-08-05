import Pkg;
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("Plots")
Pkg.add("Lathe")
Pkg.add("GLM")
Pkg.add("StatsPlots")
Pkg.add("MLBase")
Pkg.add("Missings")
Pkg.add("Statistics")
Pkg.add("ClassImbalance")
Pkg.add("ROCAnalysis")
Pkg.add("FreqTables")

using Pkg
using DataFrames
using CSV
using GLM
using Lathe
using MLBase
using ClassImbalance
using ROCAnalysis

using CSV,DataFrames
df = CSV.read("C:\\Users\\ardwived\\Downloads\\bestsellers.csv",DataFrame)
first(df,1) #for only 1 rows to display, up to you

#to check the size of the dataset
size(df)

#By default the dropmissing and dropmissing! functions keep the Union{T, Missing} element type in columns selected for row removal. To remove the Missing part, if present, set the disallowmissing option to true (it will become the default behavior in the future).
dropmissing(df)


#One Hot Encoding
target_final=Lathe.preprocess.OneHotEncode(df,:Genre)

#to see the names of all elements
names(target_final)

#Feature selection

specifics=df[:,[:User_Rating,:Reviews,:Price,:Year,:fiction]]

#Checkbalance
using FreqTables
classes=freqtable(target_final[:fiction])

#we need to reduce the number of false +ve and false -ve because they are misleading data
#split Data
using Lathe.preprocess:TrainTestSplit 
train,test =TrainTestSplit(specifics,.75)


#Build Model, to build model 1st specify formula, glm to build model
fm=@formula(fiction~User_Rating+Reviews+Price+Year)
logit=glm(fm,train,Binomial(),LogitLink())

#Prediction
predictions=predict(logit,test)

#to convert output in 0's and 1's
prediction_class=[if x<0.5 0 else 1 end for x in predictions]

#now take the result and turn it to dataframe naming the dataframe prediction_df
prediction_df=DataFrame(y_actual=test.fiction,y_predicted=prediction_class,prob_predicted=predictions)

#now we get actual false +ve and -ve 
prediction_df.correctly_classified=prediction_df.y_actual.==prediction_df.y_predicted
#(in this we are comparing the actual and predicted values, making sure they are exactly equal)

#Accuracy
accuracy =mean(prediction_df.correctly_classified)

#Confusion Matrix
#is a summary of predicting result on target variable
confusion_matrix=MLBase.roc(prediction_df.y_actual,prediction_df.y_predicted)

#False Negative Rate
false_negative_rate(confusion_matrix)

classes


x2,y2=smote(specifics[!,[:User_Rating,:Reviews,:Price,:Year]],specifics.fiction,k=1,pct_under=200,pct_over=100)

balanced_classes=freqtable(y2)

specifics_new=hcat(x2,y2)

rename!(specifics_new,:x1=>:target)

#Build New Model



#Obtain New Predictions
fm1=@formula(target~User_Rating+Reviews+Price+Year)
logit1=glm(fm1,specifics_new,Binomial(),LogitLink())

prediction1=predict(logit1,test);

prediction_class1= [if x<0.5 0 else 1 end for x in prediction1];

prediction_new=DataFrame(y_actual1=test.fiction,y_predicted1=prediction_class1,prob_predicted1=prediction1);

prediction_new.correctly_classified1=prediction_new.y_actual1 .==prediction_new.y_predicted1;

#Accuracy
accuracy_final=mean(prediction_new.correctly_classified1)

#Confusion Matrix¶
confusion_matrix_final=MLBase.roc(prediction_new.y_actual1,prediction_new.y_predicted1)

confusion_matrix

#False Negative Rate
false_negative_rate(confusion_matrix_final)

false_negative_rate(confusion_matrix)


