## THE FOLLOWING DATASET WAS FOUND ON THE UC IRVINE MACHINE LEARNING REPOSITORY
## DONATED TO UC IRVINE ON 11/26/2014
## URL FOR ACCESS: https://archive.ics.uci.edu/dataset/320/student+performance
## ACCESS DATE: 8/11/2025
## CREATED BY PAULO CORTEZ
## LICENSED UNDER A CREATIVE COMMONS ATTRIBUTION 4.0 INTERNATIONAL
## (CC BY 4.0) LICENSE
## DOWNLOAD LINK: https://archive.ics.uci.edu/static/public/320/student+performance.zip

## The following code that installs the required packages was inspired from code in the HarvardX Data Science Program
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(readxl)) install.packages("readxl", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "https://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(writexl)) install.packages("writexl", repos = "http://cran.us.r-project.org")
if(!require(utils)) install.packages("utils", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(dplyr)
library(readr)
library(readxl)
library(writexl)
library(randomForest)
library(ggplot2)
library(corrplot)
library(utils)

url <- "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
temp_dir <- tempdir()
outer_zip <- file.path(temp_dir, "student+performance.zip") ## this line has minor inspiration from GitHub Copilot
download.file(url, destfile = outer_zip, mode = "wb")

unzip(outer_zip, exdir = temp_dir)

student_zip <- list.files(temp_dir, pattern = "student.zip$", recursive = TRUE, full.names = TRUE)

unzip(student_zip, exdir = temp_dir)

csv_file <- list.files(temp_dir, pattern = "student-mat.csv$", recursive = TRUE, full.names = TRUE)

student_data <- read_csv2(csv_file)
write_xlsx(student_data, path = "student-mat.xlsx")

df <- read_excel("student-mat.xlsx")
## DATASET WAS ACCESSED FROM URL: https://archive.ics.uci.edu/dataset/320/student+performance
## DOWNLOAD LINK: https://archive.ics.uci.edu/static/public/320/student+performance.zip


## Exploratory Data Analysis
## START CLEANING THE DATA SET
## Check for NA's
sum(is.na(df)) ## There are no NA values

## Examine the dataset
str(df)
head(df)
dim(df)
names(df)

## Start off by seeing how G3 is distributed
df %>%
  ggplot(aes(G3)) +
  geom_histogram(binwidth = 1, color = "black") +
  ggtitle("G3 Distribution") +
  xlab("G3") +
  ylab("Frequency")
## Approximately normally distributed but large outlier at 0.
## Getting rid of the outlier at 0
df_new <- df %>%
  filter(G3 != 0)

## We will view the distributions for G1 and G2
df_new %>%
  ggplot(aes(G1)) +
  geom_histogram(binwidth = 1, color = "black") +
  ggtitle("G1 Distribution") +
  xlab("G1") +
  ylab("Frequency")

df_new %>%
  ggplot(aes(G2)) +
  geom_histogram(binwidth = 1, color = "black") +
  ggtitle("G2 Distribution") +
  xlab("G2") +
  ylab("Frequency")

## Both are approximately normally distributed
## Check for 0's
sum(df_new$G1 == 0)
sum(df_new$G2 == 0)
## There are no zeros or outliers

## Checking whether predictors are independent or not
numeric_grade <- df_new[, sapply(df_new, is.numeric)]
correlation_matrix <- cor(numeric_grade)
corrplot(correlation_matrix, method = "color", type = "upper", tl.cex = 1)
## Medu and Fedu highly correlated
## Dalc and Walc are highly correlated
## G1 and G2 are also highly correlated

## Splitting into our training set and final test set
set.seed(1)
train_index <- createDataPartition(df_new$G3, p = 0.9, list = FALSE)
grade <- df_new[train_index,]
final_test <- df_new[-train_index,]

## Getting familiar with the new dataset
str(grade)
head(grade)
dim(grade)
names(grade)

## We'll compare the differences of final scores between binary variables
## G3 per School
grade %>%
  ggplot(aes(school,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per school") +
  xlab("School") +
  ylab("Average G3 Score")
  ## GP has slightly higher score, but also more students

## G3 per sex
grade %>%
  ggplot(aes(sex,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per sex") +
  xlab("Sex") +
  ylab("Average G3 Score")
  ## Males score slightly higher

## G3 per address
grade %>%
  ggplot(aes(address,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per Address") +
  xlab("Address") +
  ylab("Average G3 Score")
  ## Urban households score higher

## G3 per famsize
grade %>%
  ggplot(aes(famsize,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per Family Size") +
  xlab("Family Size") +
  ylab("Average G3 Score")
  ## Nearly Equal

## G3 per Pstatus
grade %>%
  ggplot(aes(Pstatus,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per Parental Status") +
  xlab("Parental Status") +
  ylab("Average G3 Score")
## Nearly Equal

## G3 per schoolsup
grade %>%
  ggplot(aes(schoolsup,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per Educational Support") +
  xlab("School Support") +
  ylab("Average G3 Score")
## Educational Support scores lower

## G3 per famsup
grade %>%
  ggplot(aes(famsup,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per Family Educational Support") +
  xlab("Parental Status") +
  ylab("Average G3 Score")
## No Family Education Support scores higher

## G3 per Extra Paid Classes
grade %>%
  ggplot(aes(paid,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per Extra Paid Classes") +
  xlab("Extra Paid Classes") +
  ylab("Average G3 Score")
## Nearly Equal with the 'no' slightly higher

## G3 per Extracurricular
grade %>%
  ggplot(aes(activities,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per Extracurriculars") +
  xlab("Extracurriculars") +
  ylab("Average G3 Score")
## Extracurriculars score slightly higher

## G3 per nursery school
grade %>%
  ggplot(aes(nursery,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per Nursery School") +
  xlab("Nursery School") +
  ylab("Average G3 Score")
## Nursery School students score slightly higher

## G3 per will for higher education
grade %>%
  ggplot(aes(Pstatus,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per will for higher education") +
  xlab("Wants Higher Education") +
  ylab("Average G3 Score")
## Nearly Equal

## G3 per internet access
grade %>%
  ggplot(aes(internet,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per Internet Access") +
  xlab("Internet") +
  ylab("Average G3 Score")
## With Internet Access scores slightly higher

## G3 per romantic status
grade %>%
  ggplot(aes(Pstatus,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per Romantic Status") +
  xlab("In Relationship") +
  ylab("Average G3 Score")
## Nearly equal

## Try to see trends in categorical variables
## G3 per Mother's job
grade %>%
  ggplot(aes(Mjob,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per Mother's Job") +
  xlab("Mother's Job") +
  ylab("Average G3 Score")
## Health and Services score slightly higher than the rest

## G3 per Father's job
grade %>%
  ggplot(aes(Fjob,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per Father's Job") +
  xlab("Father's Job") +
  ylab("Average G3 Score")
## Mainly equal except for people with teachers as dads

## G3 per Reason for Study
grade %>%
  ggplot(aes(reason,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per Reason for Study") +
  xlab("Reason for Study") +
  ylab("Average G3 Score")
## Nearly Equal

## G3 per Guardian
grade %>%
  ggplot(aes(guardian,G3)) +
  stat_summary(fun = "mean", geom = "bar") +
  ggtitle("G3 Score per Guardian") +
  xlab("Guardian") +
  ylab("Average G3 Score")
## People with a dad score slightly higher than others

## PARTITIONING DATA INTO TRAINING AND TEST AGAIN
set.seed(1)
test_index <- createDataPartition(grade$G3, times=1, p=0.2, list=FALSE)
test_set <- grade[test_index,]
training_set <- grade[-test_index,] #80% of the data used for the training set

## BEGINNING OF CREATING LINEAR MODEL WITH G1 AND G2
## Define RMSE function
rmse <- function(actual, predicted){
  sqrt(mean((actual - predicted)^2))
}

mu <- mean(training_set$G3) ## represents the global score
mu ## mu is 11.55469

rmse(mu, training_set$G3) ## 3.182738

## Creating a function to get predictor's bias
bias <- function(column_name){
  training_set %>%
    group_by(!!sym(column_name)) %>%
    summarize(bias=mean(G3 - mu, na.rm = TRUE))
}

## Creating a prediction function now that we have the bias
predicted <- function(bias, column_name){
  test_set %>%
    left_join(bias, by = column_name) %>%
    mutate(prediction = mu + replace_na(bias,0)) %>%
    pull(prediction)
}

grade_rmse <- function(column_name){
  bias <- bias(column_name)
  predicted <- predicted(bias, column_name)
  rmse(predicted,test_set$G3)
}

grade_rmse('G1') ## 1.611524
grade_rmse('G2') ## 0.8607571
## both heavily improved RMSE

G1_bias <- bias('G1')
G1_G2_bias <- training_set %>%
  left_join(G1_bias, by = 'G1') %>%
  group_by(G2) %>%
  summarize(G2_bias=mean(G3-mu-bias, na.rm = TRUE))

prediction <- test_set %>%
  left_join(G1_bias, by = 'G1') %>%
  left_join(G1_G2_bias, by = 'G2') %>%
  mutate(prediction=mu+bias+G2_bias) %>%
  pull(prediction)
rmse(prediction, test_set$G3) ## 1.485441

## Use regularization
lambdas <- seq(0,10,0.5)
lambda_rmse <- sapply(lambdas, function(lambda){
  G1_bias <- training_set %>%
    group_by(G1) %>%
    summarize(bias = sum(G3-mu)/(n()+lambda))
  
  G2_bias <- training_set %>%
    left_join(G1_bias, by = "G1") %>%
    group_by(G2) %>%
    summarize(G2_bias = sum(G3-mu-bias)/(n()+lambda))
  
  prediction <- test_set %>%
    left_join(G1_bias, by = 'G1') %>%
    left_join(G2_bias, by = 'G2') %>%
    mutate(
      bias = replace_na(bias,0),
      G2_bias = replace_na(G2_bias,0),
      prediction = mu+bias+G2_bias) %>%
    pull(prediction)
  rmse(prediction,test_set$G3)
})

plot(lambdas, lambda_rmse)

## Calculating best lambda
lambda <- lambdas[which.min(lambda_rmse)]
lambda ## best lambda is 2

## Final RMSE Calculation
final_mu <- mean(grade$G3)
## Final bias Calculations
final_G1_bias <- grade %>%
  group_by(G1) %>%
  summarize(bias = sum(G3-final_mu)/(n()+lambda))

final_G1_G2_bias <- grade %>%
  left_join(final_G1_bias, by = 'G1') %>%
  group_by(G2) %>%
  summarize(G2_bias=sum(G3 - final_mu - bias)/(n()+lambda))

##Final Prediction
final_prediction <- final_test %>%
  left_join(final_G1_bias, by = 'G1') %>%
  left_join(final_G1_G2_bias, by = 'G2') %>%
  mutate(
    bias=replace_na(bias, 0),
    G2_bias = replace_na(G2_bias,0),
    prediction = final_mu+bias+G2_bias) %>%
  pull(prediction)
rmse(final_prediction,final_test$G3) #Final RMSE is 1.138002

## The structure for the random forest models were inspired from examples in the HarvardX Data Science course
## Random Forest Model WITHOUT G1 and G2
## Get rid of highly correlated variables
highly_correlated <- c("G1", "G2")
grade_new <- grade %>%
  select(-all_of(highly_correlated)) ## minor inspiration from GitHub Copilot
final_test_new <- final_test %>%
  select(-all_of(highly_correlated)) ## minor inspiration from GitHub Copilot

## Set up Random Forest Training Model
rf_model <- train(
  G3 ~ .,
  data = grade_new,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = data.frame(mtry = 1:10)
)

best_mtry <- rf_model$bestTune$mtry
best_mtry ## best_mtry is 9

final_rf_model <- randomForest(
  G3 ~ .,
  data = grade_new,
  mtry = best_mtry,
  ntree = 500,
  importance = TRUE
)

rf_predicted <- predict(final_rf_model, newdata = final_test_new)
rf_rmse <- rmse(final_test_new$G3, rf_predicted)
rf_rmse ## RMSE was 3.171029

## Now we will do a random forest prediction of all predictors
rf_model_all <- train(
  G3 ~ .,
  data = grade,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = data.frame(mtry = 1:10)
)

final_best_mtry <- rf_model_all$bestTune$mtry
final_best_mtry ## final_best_mtry is 10

final_rf_model_all <- randomForest(
  G3 ~ .,
  data = grade,
  mtry = final_best_mtry,
  ntree = 500,
  importance = TRUE
)

rf_predicted <- predict(final_rf_model_all, newdata = final_test)
rf_rmse <- rmse(final_test$G3, rf_predicted)
rf_rmse ## RMSE was 0.9568145