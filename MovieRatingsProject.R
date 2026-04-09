##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1) # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#getting familiar with edx
head(edx)
dim(edx)
names(edx)

edx %>% select(movieId, title) %>% head(10) # see that movieId and title represent the same movie

dim(final_holdout_test) #see the dimensions of final_holdout_test

#analyzing the relationships between predictors and ratings
#simplifying the timestamps variable to year
#creating new edx data set with year from timestamp
edx_new <- edx %>%
  mutate(year=year(as_datetime(timestamp)))

edx_new %>%
  ggplot(aes(year)) +
  geom_histogram(binwidth = 1, color = "black") +
  ggtitle("Year vs. Number of Ratings") +
  xlab("Year") +
  ylab("# Ratings")

#movieId and ratings
length(unique(edx_new$movieId)) # 10677 observations
edx_new %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins= 30, color = "black") +
  scale_x_log10() +
  ggtitle("Movie ID vs. Number of Ratings") +
  xlab("Movie ID") +
  ylab("# Ratings")

#userId and ratings
length(unique(edx_new$userId)) # 69878 observations
edx_new %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins=30, color="black") +
  scale_x_log10() +
  ggtitle("User ID vs. Number of Ratings") +
  xlab("User ID") +
  ylab("# Ratings")

#genres and ratings
length(unique(edx_new$genres)) #797 observations
edx_new %>% count(genres) %>%
  ggplot(aes(n)) +
  geom_histogram(bins=30, color= "black") +
  scale_x_log10() +
  ggtitle("Genres vs. Number of Ratings") +
  xlab("Genres") +
  ylab("# Ratings")

#distribution of ratings
length(unique(edx_new$rating)) #10 observations
edx_new %>% 
  ggplot(aes(rating)) +
  geom_histogram(color = "black") +
  ggtitle("Ratings") +
  xlab("Rating") +
  ylab("# Ratings")

#Partition 20% of the data for the testing set
set.seed(1) #set the seed for reproducibility
test_index <- createDataPartition(edx_new$rating, times=1, p=0.2, list=FALSE)
test_set <- edx_new[test_index,]
training_set <- edx_new[-test_index,] #80% of the data used for the training set

#defining the RMSE function
rmse <- function(actual, predicted){
  sqrt(mean((actual - predicted)^2))
}

mu <- mean(training_set$rating) #represents the global average rating
mu

rmse(mu, test_set$rating)

#creating a function to get the bias
movie_bias <- function(column_name){
  training_set %>%
    group_by(!!sym(column_name)) %>%
    summarize(bias=mean(rating - mu, na.rm = TRUE))
}

#creating a prediction function now that we have the bias
movie_predicted <- function(movie_bias, column_name){
  test_set %>%
    left_join(movie_bias, by = column_name) %>%
    mutate(prediction = mu + replace_na(bias,0)) %>%
    pull(prediction)
}

movie_rmse <- function(column_name){
  bias <- movie_bias(column_name)
  predicted <- movie_predicted(bias, column_name)
  rmse(predicted,test_set$rating)
}

movie_rmse('movieId') # approximately normally distributed so the RMSE is lower
movie_rmse('userId') # approximately normally distributed so the RMSE is lower
movie_rmse('genres') # not normally distributed so the RMSE is higher
movie_rmse('year') # not normally distributed so the RMSE is higher

#combining different biases for lower RMSE
#movieId, userId, and genres improved RMSE
movieId_bias <- movie_bias('movieId')
#combining movieId and userId bias
movie_user_bias <- training_set %>%
  left_join(movieId_bias, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(user_bias=mean(rating-mu-bias, na.rm = TRUE))
#combining movieId, userId, and genre bias
all_bias <- training_set %>%
  left_join(movieId_bias, by="movieId") %>%
  left_join(movie_user_bias, by="userId") %>%
  group_by(genres) %>%
  summarize(genre_bias = mean(rating-mu-bias-user_bias,na.rm=TRUE))

#adding the combined biases to our prediction
prediction <- test_set %>%
  left_join(movieId_bias, by = 'movieId') %>%
  left_join(movie_user_bias, by = 'userId') %>%
  left_join(all_bias, by = 'genres') %>%
  mutate(
    bias = replace_na(bias,0),
    user_bias = replace_na(user_bias,0),
    genre_bias = replace_na(genre_bias,0),
    prediction = mu+bias+user_bias+genre_bias) %>%
  pull(prediction)
rmse(prediction,test_set$rating) #calculating the new RMSE with the combined biases
#RMSE is greatly improved with combined biases

lambdas <- seq(0, 10, 0.5)
lambda_rmse <- sapply(lambdas,function(lambda){
  
  movieId_bias <- training_set %>%
    group_by(movieId) %>%
    summarize(bias = sum(rating-mu)/(n()+lambda))
  
  movie_user_bias <- training_set %>%
    left_join(movieId_bias, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(user_bias=sum(rating - mu - bias)/(n()+lambda))
  
  all_bias <- training_set %>%
    left_join(movieId_bias, by="movieId") %>%
    left_join(movie_user_bias, by="userId") %>%
    group_by(genres) %>%
    summarize(genre_bias = sum(rating - mu - bias - user_bias)/(n()+lambda))
  
  prediction <- test_set %>%
    left_join(movieId_bias, by = 'movieId') %>%
    left_join(movie_user_bias, by = 'userId') %>%
    left_join(all_bias, by = 'genres') %>%
    mutate(
      bias = replace_na(bias,0),
      user_bias = replace_na(user_bias,0),
      genre_bias = replace_na(genre_bias,0),
      prediction = mu+bias+user_bias+genre_bias) %>%
    pull(prediction)
  rmse(prediction,test_set$rating)
})
plot(lambdas, lambda_rmse)

#getting the best lambda that gets us our lowest RMSE
lambda <- lambdas[which.min(lambda_rmse)]
lambda # best lambda is 5

#beginning test on final holdout test set
final_mu <- mean(edx$rating) #now we grab the mu of the edx data set to test on the final holdout set
#combine the biases one more time
final_movieId_bias <- edx %>%
  group_by(movieId) %>%
  summarize(bias = sum(rating-final_mu)/(n()+lambda))

final_movie_user_bias <- edx %>%
  left_join(final_movieId_bias, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(user_bias=sum(rating - final_mu - bias)/(n()+lambda))

final_all_bias <- edx %>%
  left_join(final_movieId_bias, by="movieId") %>%
  left_join(final_movie_user_bias, by="userId") %>%
  group_by(genres) %>%
  summarize(genre_bias = sum(rating - final_mu - bias - user_bias)/(n()+lambda))

prediction <- final_holdout_test %>%
  left_join(final_movieId_bias, by = 'movieId') %>%
  left_join(final_movie_user_bias, by = 'userId') %>%
  left_join(final_all_bias, by = 'genres') %>%
  mutate(
    bias=replace_na(bias, 0),
    user_bias = replace_na(user_bias,0),
    genre_bias = replace_na(genre_bias,0),
    prediction = final_mu+bias+user_bias+genre_bias) %>%
  pull(prediction)
rmse(prediction,final_holdout_test$rating) #Final RMSE is 0.8646798
