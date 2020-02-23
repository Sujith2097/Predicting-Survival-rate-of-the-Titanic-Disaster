# Predicting-Survival-rate-of-the-Titanic-Disaster
The main goal of the project is to predict the survival rate of the passengers by using a machine learning model and analyze what sort of people have survived based on the features like name, age, gender and passenger class. Apache spark frame work is used in this project due to its accuracy in handling the data. A machine learning model, Random forest classifier is used in the data analytical approach and Apache framework is used for its exceptional accuracy in the data handling. 
# Data Description
The data set used in the research for the predicting the survival of titanic disaster is provided by the dataset platform kaggle. The data set is comprised of 891 rows in the train dataset and 418 rows in the test dataset. The dataset contains certain attributes for every passengers like PassengerID, name of the passenger, Class of the passenger, age, sex, number of spouse or siblings on board, parents or children on board, port of embarkation where they have boarded the ship. The dataset is provided in the form of a CSV file (comma separated file).
# scope
The main goal of the project is to predict the survival rate of the passengers by using a machine learning model and analyze what sort of people have survived based on the features like name, age, gender and passenger class and Checking the accuracy of the model by using the machine learning model.
# Frameworks and Libraries used:
1) Apache spark
2) Multiclass classification evaluator
3) Vector assembler 
4) String Indexer
5) Numpy 
6) Pandas
# Solution Design
To obtain the solution for the problem stated for the analysis we need to first understand the data. In this project we need train dataset is used for training with the machine learning model and we need to understand the problem statement and check whether which attribute is effecting the possible outcomes. Our goal is to predict the count of survival of the passengers in the ship. In our project sex, age, passenger class are the important attributes as they have the capability of giving wrong prediction. In order overcome this data cleaning and preprocessing is done before applying any machine learning model.
# Data Cleaning 
The data cleaning and analysis is started with exploration of speciﬁc features available in the dataset. The age and cabin columns most of the NA values in them. As age has 177 null values and drop the cabin column as it has more null values and which is not an important attribute. Initials such as Ms or Mme stand for Miss are misplaced. I have substituted them for Miss and the same for certain values. Embarked attribute has only two missing values and which is dropped as it is a trivial attribute.
# Feature Engineering
Initially, it chooses the attributes that are helpful in the prediction Since several functionalities are utilized for output prediction, we have used the vector assembler and QuantileDiscretizer function to extract and transform specific functions like Pclass, age, Ticket fare, parch(parent and children, sibsp(siblings and spouse) into a single vector column. ⠀The single column vector can be used in giving the initial input attributes to the model.
# Modelling
Various models can be used to address the problem like logistic regression, Naive Bayes classifier, decision tree, Random forest classifier and support vector machine. To choose from the above algorithms decision tree and random forest classifier will be perfect fit for the problem as they have the potentiality of the Multiclass classification. ⠀In this model we have used various features were taken like passenger class, age, ParCh and SibSp. All these multiple features are converted in single column vector by using a vector assembler to give input the random forest classifier. The trees in the internal nodes are built by the algorithm to give the output.
# Performance Evaluation
To test the accuracy we have used the ROC curve as we have got the area under the ROC as 1 which means the each class is predicted accurately without any error and a predictive ML model(Random Forest classifier is used with the accuracy outcome of 0.83 and a test error of 0.16 which is highly interpretable.
# Visualization
In the visualization we have shown the comparision of Survival to the fatality rate by plotting in a bar graph. Next we have compare the survival rate based on the feature sex group in that we can see that more number of females survived when compared to males. The next attribute we compared is the passenger class in which Upper class males survived comapred to the lower class. 
