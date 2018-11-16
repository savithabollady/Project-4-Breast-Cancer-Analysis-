# Project-4-Breast-Cancer-Analysis-
Find my detail analysis here
https://docs.google.com/presentation/d/1Z35ydQhHP3QQFYQg8HLXQoGroEUHTQ0Pyi_0NjnqdfE/edit?usp=sharing


Code here:


#Import dataset
wbcd1 <- read.csv("C:/Users/Savitha/Desktop/R Studio/Project to Submit/Project_files_DART4.1-20181029T040815Z-001/Project_files_DART4.1/CancerData.csv",header=T, stringsAsFactors=F)
View(wbcd1)
wbcd1$X <- NULL
wbcd1<- wbcd1[-1]
#Reshape the datasets

wbcd1$diagnosis <- factor(ifelse(wbcd1$diagnosis=="B","Benign","Malignant"))
table(wbcd1$diagnosis)
text(barplot(table(wbcd1$diagnosis), col = c('green', 'red'),
             main = 'Bar Plot of Diagnosis'), 0, 
     table(wbcd1$diagnosis), cex = 2, pos = 3)
str(wbcd1)
summary(wbcd1)
knitr::kable(head(wbcd1))
head(wbcd1)
dim(wbcd1)
variable.names(wbcd1)

#propotion
round(prop.table(table(wbcd1$diagnosis))*100,digits = 2)

#3. Analyze the Correlation between variables
#Mean
library(PerformanceAnalytics)
chart.Correlation(wbcd1[,c(2:11)],histogram=TRUE, col="grey10", pch=1, main="Cancer Mean")

#See the relation between each variables (diagnosis included)

library(ggplot2)
library(GGally)
library(stringi)
ggpairs(wbcd1[,c(2:11,1)], aes(color=diagnosis, alpha=0.75), 
        lower=list(continuous="smooth"))+ theme_bw()+
  labs(title="Cancer Mean")+
  theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))

ggcorr(wbcd1[,c(2:11)], name = "corr", label = TRUE)+
  theme(legend.position="none")+
  labs(title="Cancer Mean")+
  theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))


# 4. Principal Component Analysis (PCA)
#Too many variables can cause such problems below
#Increased computer throughput
#Too complex visualization problems
#Decrease efficiency by including variables that have no effect on the analysis
#Make data interpretation difficult
#If you see the ggcorr plot above[3-3], high correlation value means it has "multicollinearity" between variables.
#-> Use one main component for model development by reduct the variables with high correlation.
#**When determining the number of principal components, use the cumulative contribution rate or use a screeplot and use the previous step of the principal component where the eigenvalue curve lies horizontally.
#PCA uses standardized data so that it can avoid data distortion caused by scale difference.
 library(factoextra)
wbcd_pca <- transform(wbcd1) 
wbcd_pca
#4-1) Summary
#In the results of PCA, if the cumulative proportion is 85% or above, it can be determined by the number of principal components.

#==================#View Point : Cumulative Proportion#=================#

#The cumulative proportion from PC1 to PC6 is about 88.7%. (above 85%)
#It means that PC1~PC6 can explain 88.7% of the whole data.
library(corrplot)
library(corrgram)
library(pcaPP)
all_pca<-prcomp(wbcd_pca[,c(2:31)],scale = TRUE)
summary(all_pca)

mean_pca <- prcomp(wbcd_pca[,c(2:11)], scale = TRUE)
summary(mean_pca)
se_pca <- prcomp(wbcd_pca[,c(12:21)], scale = TRUE)
summary(se_pca)

worst_pca <- prcomp(wbcd_pca[,c(22:31)], scale = TRUE)
summary(worst_pca)

#The percentage of variability explained by the principal components can 
#be ascertained through screeplot.
#=> View Point : principal components where the line lies.
#All: Line lies at point PC6
fviz_eig(all_pca, addlabels=TRUE, ylim=c(0,60), geom = c("bar", "line"), 
         barfill = "pink", barcolor="grey",linecolor = "red", ncp=10)+
  labs(title = "Cancer All Variances - PCA",
       x = "Principal Components", y = "% of variances")


#Get PCA Variables for All Variables
all_var <- get_pca_var(all_pca)
all_var
# Correlation between variables and PCA
library("corrplot")
corrplot(all_var$cos2, is.corr=FALSE)

#Contributions of variables to PCA
#To highlight the most contributing variables for each components
corrplot(all_var$contrib, is.corr=FALSE)    

mean_var <- get_pca_var(mean_pca)
mean_var

##Quality of representation of PCA
#Correlation between variables and PCA
corrplot(mean_var$cos2, is.corr=FALSE)
#To highlight the most contributing variables for each components
corrplot(mean_var$contrib, is.corr=FALSE)   



#Contributions of variables to PC1 & PC2
library(gridExtra)
p1 <- fviz_contrib(worst_pca, choice="var", axes=1, fill="pink", color="grey", top=10)
p2 <- fviz_contrib(worst_pca, choice="var", axes=2, fill="skyblue", color="grey", top=10)
grid.arrange(p1,p2,ncol=2)

#Cluster Analysis- See the plot - color variables by groups
#value centers : put the optimal principal component value that we chosen above.
set.seed(218)
res.all <- kmeans(all_var$coord, centers = 6, nstart = 25)
grp <- as.factor(res.all$cluster)

fviz_pca_var(all_pca, col.var = grp, 
             palette = "jco",
             legend.title = "Cluster")

fviz_pca_biplot(all_pca, col.ind = wbcd1$diagnosis, col="black",
                palette = "jco", geom = "point", repel=TRUE,
                legend.title="Diagnosis", addEllipses = TRUE)

#============Apply ML methods and compare each other and choose best fits===============#

#1) Make test & train dataset for testing classification ML methods
#Shuffle the wbcd data(100%) & Make train dataset(70%), test dataset(30%)
nrows <- NROW(wbcd1)
set.seed(218)                           ## fix random value
index <- sample(1:nrows, 0.7 * nrows)   ## shuffle and divide

#train <- wbcd                          ## 569 test data (100%)
train <- wbcd1[index,]                   ## 398 test data (70%)
test <- wbcd1[-index,]                   ## 171 test data (30%)

#2) Check the proportion of diagnosis (Benign / Malignant)
prop.table(table(train$diagnosis))
prop.table(table(test$diagnosis))



#3) Apply every ML methodsto data
#Precission Method
install.packages("caret", dependencies = c("Depends", "Suggests"))
install.packages("lattice")
library(lattice)
library(caret)
library(robustbase)
library(tree)
library(tidyr)
library(C50)
library(yaml)
library(libcoin)

library(rpart)
learn_rp <- rpart(diagnosis~.,data=train,control=rpart.control(minsplit=2))
pre_rp <- predict(learn_rp, test[,-1], type="class")
cm_rp  <- confusionMatrix(pre_rp, test$diagnosis)   
cm_rp

#Prune
learn_pru <- prune(learn_rp, cp=learn_rp$cptable[which.min(learn_rp$cptable[,"xerror"]),"CP"])
pre_pru <- predict(learn_pru, test[,-1], type="class")
cm_pru <-confusionMatrix(pre_pru, test$diagnosis)           
cm_pru

#Random Forest
library(randomForest)
library(ggplot2)
learn_rf <- randomForest(diagnosis~., data=train,  mtry=10, proximity=T, importance=T)
pre_rf   <- predict(learn_rf, test[,-1])
cm_rf    <- confusionMatrix(pre_rf, test$diagnosis)
cm_rf
plot(learn_rf, main="Random Forest (Error Rate vs. Number of Trees)")


##### naiveBayes without laplace
learn_nb <- naiveBayes(train[,-1], train$diagnosis)
pre_nb <- predict(learn_nb, test[,-1])
cm_nb <- confusionMatrix(pre_nb, test$diagnosis)		
cm_nb

##### Prediction Plot
plot(margin(learn_rf,test$diagnosis))

##### Variance Importance Plot
#- MeanDecreaseAccuracy : radius_worst > concave.points_worst > area_worst > perimeter_worst
#Important parameters for accuracy improvement are determined by the "MeanDecreaseAccuracy".
# MeanDecreaseGini : perimeter_worst > radius_worst > area_worst > concave.points_worst

#Important parameters for improving node impurities are determined by the "MeanDecreaseGini".
```{r}
varImpPlot(learn_rf)
```



#### ctree

library(party)
learn_ct <- ctree(diagnosis~., data=train, controls=ctree_control(maxdepth=2))
pre_ct   <- predict(learn_ct, test[,-1])
cm_ct    <- confusionMatrix(pre_ct, test$diagnosis)
cm_ct


#Adaptive Boosting
library(rpart)
library(ada)
control <- rpart.control(cp = -1, maxdepth = 14,maxcompete = 1,xval = 0)
learn_ada <- ada(diagnosis~., data = train, test.x = train[,-1], test.y = train[,1], type = "gentle", control = control, iter = 70)
pre_ada <- predict(learn_ada, test[,-1])
cm_ada <- confusionMatrix(pre_ada, test$diagnosis)
cm_ada

# Support Vector Machine

learn_svm <- svm(diagnosis~., data=train)
pre_svm <- predict(learn_svm, test[,-1])
cm_svm <- confusionMatrix(pre_svm, test$diagnosis)
cm_svm

#Visualize to compare the accuracy of all methods
col <- c("#ed3b3b", "#0099ff")
par(mfrow=c(3,5))
fourfoldplot(cm_rp$table, color = col, conf.level = 0, margin = 1, main=paste("RPart (",round(cm_rp$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_pru$table, color = col, conf.level = 0, margin = 1, main=paste("Prune (",round(cm_pru$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_ct$table, color = col, conf.level = 0, margin = 1, main=paste("CTree (",round(cm_ct$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_nb$table, color = col, conf.level = 0, margin = 1, main=paste("NaiveBayes (",round(cm_nb$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_knn$table, color = col, conf.level = 0, margin = 1, main=paste("Tune KNN (",round(cm_knn$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_rf$table, color = col, conf.level = 0, margin = 1, main=paste("RandomForest (",round(cm_rf$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_ada$table, color = col, conf.level = 0, margin = 1, main=paste("AdaBoost (",round(cm_ada$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_svm$table, color = col, conf.level = 0, margin = 1, main=paste("SVM (",round(cm_svm$overall[1]*100),"%)",sep=""))

# Select a best prediction model according to high accuracy

opt_predict <- c(cm_rp$overall[1], cm_pru$overall[1], cm_ct$overall[1], cm_nb$overall[1], cm_knn$overall[1],  cm_rf$overall[1], cm_ada$overall[1], cm_svm$overall[1])
names(opt_predict) <- c("rpart","prune","ctree","nb","knn","rf","ada","svm")
best_predict_model <- subset(opt_predict, opt_predict==max(opt_predict))
best_predict_model

