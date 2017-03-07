#Supervised analysis-Naive Bayes Classifier

library(NLP)
library(tm)
library(plyr)
library(ggplot2)
library(e1071)
library(topicmodels)
library(caret)
library(RTextTools)

set.seed(123)
reuters <- read.table("r8-train-all-terms.txt", header = F, sep = '\t')

reut.rand <- reuters[sample(1:nrow(reuters)),]
reut.rand <- reut.rand[which(reut.rand$V1 %in% c("trade","crude","money-fx")), ]
corpus <- Corpus(VectorSource(reut.rand$V2))

corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removeWords, stopwords("english"))

mat <- DocumentTermMatrix(corpus)

mat4 <- weightTfIdf(mat)
mat4 <- as.matrix(mat4)

classifier <- naiveBayes(mat4[1:568,],reut.rand$V1[1:568])
predicted <- predict(classifier, newdata = mat4[569:710,])
table(as.character(reut.rand$V1[569:710]), as.character(predicted))
# in rtexttools library
recall_accuracy(as.character(reut.rand$V1[569:710]), as.character(predicted))


# tree

container <- create_container(mat, reut.rand$V1, trainSize = 1:568, testSize = 569:710, virgin = F)
model  <- train_model(container, "TREE", kernel = "linear")
results <- classify_model(container, model)
table(as.character(reut.rand$V1[569:710]), as.character(results[,"TREE_LABEL"]))
recall_accuracy(as.character(reut.rand$V1[569:710]), as.character(results[,"TREE_LABEL"]))


#svm
container1 <- create_container(mat, reut.rand$V1, trainSize = 1:568, testSize = 569:710, virgin = F)
modelsvm <- train_model(container1, "SVM", kernel = "linear")
results_svm <- classify_model(container1, modelsvm)
table(as.character(reut.rand$V1[569:710]), as.character(results_svm[,"SVM_LABEL"]))
recall_accuracy(as.character(reut.rand$V1[569:710]), as.character(results_svm[,"SVM_LABEL"]))