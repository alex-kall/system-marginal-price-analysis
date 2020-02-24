library(lubridate)
library(DT)
library(forecast)
library(plyr)
library(ggplot2)
library(nnet)
library(neuralnet)
library(RSNNS)
library(data.table)
library(Metrics)
library(randomForest)
library(gbm)

# Read data
train_all <- read.csv("TrainSet.csv", stringsAsFactors=F, sep=",")
train_new <- head(train_all, length(train_all) - 143)
train <- head(train_new, length(train_new) - 168)
test <- tail(train_new, 168)

# See data
datatable(tail(train, 100))
autoplot(ts(train$Prices.BE, frequency=24), ylab="Price BE")

# See data - reasonable scale
autoplot(ts(train$Prices.BE, frequency=24), ylab="Price BE", ylim=c(0, 200))

# See data - last month
autoplot(ts(tail(train$Prices.BE, 30 * 24), frequency=24), ylab="Price BE", ylim=c(0, 200))

# Identify missing values
missing <- train[is.na(train$Prices.BE) == T,]
datatable(missing)
nrow(missing)

# Fix missing values
train$Date <- as.Date(train$datetime_utc)
train$Year <- year(train$Date)
train$Month <- month(train$Date)
train$DateType <- wday(train$Date)
train$Hour <- hour(train$datetime_utc)
test$Date <- as.Date(test$datetime_utc)
test$Year <- year(test$Date)
test$Month <- month(test$Date)
test$DateType <- wday(test$Date)
test$Hour <- hour(test$datetime_utc)
train1 = train2 = train3 = train4 <- train

# Create profiles
profiles1 <- ddply(na.omit(train[, -1]), .(DateType, Hour), colwise(mean))

# Deal with extreme values
par(mfrow=c(1, 2))
boxplot(train$Prices.BE, main="Prices BE") 
plot(density(train$Prices.BE), main="Prices BE")

# Fix upper and lower bounds
LimitUp <- quantile(train$Prices.BE, 0.999)
LimitDown <- quantile(train$Prices.BE, 0.001)
train[train$Prices.BE > LimitUp,]$Prices.BE <- LimitUp
train[train$Prices.BE < LimitDown,]$Prices.BE <- LimitDown

# See data again
autoplot(ts(train$Prices.BE, frequency=24), ylab="Price BE")
par(mfrow=c(1, 2))
boxplot(train$Prices.BE, main="Prices BE") ; 
plot(density(train$Prices.BE), main="Prices BE")

# Start analysis to proceed with forecasting
par(mfrow=c(3, 3)) 
maxi <- max(train[(train$Date >= "2015-02-02") & (train$Date <= "2015-02-08"),]$Prices.BE)
mini <- min(train[(train$Date >= "2015-02-02") & (train$Date <= "2015-02-08"),]$Prices.BE)
plot(train[train$Date == "2015-02-02",]$Prices.BE,
	 main="Monday", type="l", ylab="Price", xlab="Time", ylim=c(mini, maxi))
plot(train[train$Date == "2015-02-03",]$Prices.BE,
	 main="Tuesday", type="l", ylab="Price", xlab="Time", ylim=c(mini, maxi))
plot(train[train$Date == "2015-02-04",]$Prices.BE,
	 main="Wednesday", type="l", ylab="Price", xlab="Time", ylim=c(mini, maxi))
plot(train[train$Date == "2015-02-05",]$Prices.BE,
	 main="Thursday", type="l", ylab="Price", xlab="Time", ylim=c(mini, maxi))
plot(train[train$Date == "2015-02-06",]$Prices.BE,
	 main="Friday", type="l", ylab="Price", xlab="Time", ylim=c(mini, maxi))
plot(train[train$Date == "2015-02-07",]$Prices.BE,
	 main="Saturday", type="l", ylab="Price", xlab="Time", ylim=c(mini, maxi))
plot(train[train$Date == "2015-02-08",]$Prices.BE,
	 main="Sunday", type="l", ylab="Price", xlab="Time", ylim=c(mini, maxi))

# Inspect Seasonplots
ggseasonplot(ts(train$Prices.BE, frequency=24), continuous=TRUE)
ggseasonplot(ts(train$Prices.BE, frequency=168), continuous=TRUE)

# Examine possible scenarios
timeseries <- ts(train$Prices.BE, frequency=168)
test_timeseries <- ts(test$Prices.BE, frequency=168)
fh <- 168
insample <- timeseries
outsample <- test_timeseries

# Decomposition
dec <- decompose(insample, type="additive")
plot(dec)
autoplot(head(dec$seasonal,168), ylab="Additive Seasonality")
dec <- decompose(insample, type="multiplicative")
plot(dec)
autoplot(head(dec$seasonal,168), ylab="Multiplicative Seasonality")

# Test various Forecasting Methods 
Evaluation <- data.frame(matrix(NA, ncol=1, nrow=11))
row.names(Evaluation) <- c("Naive", "SES", "sNaive", "SES_Add", "SES_Mul", "MLR", "NN", "HOLT",
						   "Random Forest", "GBM", "Comb")
colnames(Evaluation) <- c("sMAPE")

# Naive
frc1 <- naive(insample, h=fh)$mean 
Evaluation$sMAPE[1] <- mean(200 * abs(outsample - frc1) / (abs(outsample) + abs(frc1)))

# SES - no decomposition
frc2 <- ses(insample, h=fh)$mean 
Evaluation$sMAPE[2] <- mean(200 * abs(outsample - frc2) / (abs(outsample) + abs(frc2)))

# Seasonal Naive
frc3 <- as.numeric(tail(insample, fh)) + outsample - outsample
Evaluation$sMAPE[3] <- mean(200 * abs(outsample - frc3) / (abs(outsample) + abs(frc3)))

# SES - with decomposition (Additive)
Indexes_in <- decompose(insample, type="additive")$seasonal
Indexes_out <- as.numeric(tail(Indexes_in, fh))
frc4 <- ses(insample - Indexes_in, h=fh)$mean + Indexes_out
Evaluation$sMAPE[4] <- mean(200 * abs(outsample - frc4) / (abs(outsample) + abs(frc4)))

# SES - with decomposition (Multiplicative)
Indexes_in <- decompose(insample, type="multiplicative")$seasonal
Indexes_out <- as.numeric(tail(Indexes_in, fh))
frc5 <- ses(insample / Indexes_in, h=fh)$mean * Indexes_out
Evaluation$sMAPE[5] <- mean(200 * abs(outsample - frc5) / (abs(outsample) + abs(frc5)))

# Inspect results
plot(insample)
lines(frc1, col=2); 
lines(frc2, col=3);
lines(frc3, col=4); 
lines(frc4, col=5);
lines(frc5, col=6);
legend("topleft",
	   legend=c("Naive", "SES", "sNaive", "SES_Add", "SES_Mul"), col=c(2:6), lty=1, cex=0.8)

# MLR
Data_ml <- train
Data_ml$Year <- year(Data_ml$datetime_utc)  # Define Year
Data_ml$Month <- month(Data_ml$datetime_utc)  # Define Month
Data_ml$DateType <- wday(Data_ml$datetime_utc)  # Define Day
Data_ml$Weekday <- 1
Data_ml[(Data_ml$DateType == 1)|(Data_ml$DateType == 7),]$Weekday <- 0
Data_ml$Lag24 = Data_ml$Lag168 = Data_ml$Lag336 = Data_ml$Lag504 = Data_ml$Lag672 <- NA
Data_ml$Lag840 = Data_ml$Lag1008 = Lag1176 = Lag1344 <- NA  # Define Level
Data_ml$Lag24 <- head(c(rep(NA, 24), head(Data_ml, nrow(Data_ml) - 24)$Prices.BE), nrow(Data_ml))
Data_ml$Lag168 <- head(c(rep(NA, 168),
					   head(Data_ml, nrow(Data_ml) - 168)$Prices.BE), nrow(Data_ml))
Data_ml$Lag336 <- head(c(rep(NA, 336),
					   head(Data_ml, nrow(Data_ml) - 336)$Prices.BE), nrow(Data_ml))
Data_ml$Lag504 <- head(c(rep(NA, 504),
					   head(Data_ml, nrow(Data_ml) - 504)$Prices.BE), nrow(Data_ml))
Data_ml$Lag672 <- head(c(rep(NA, 672),
					   head(Data_ml, nrow(Data_ml) - 672)$Prices.BE), nrow(Data_ml))
Data_ml$Lag840 <- head(c(rep(NA, 840),
					   head(Data_ml, nrow(Data_ml) - 840)$Prices.BE), nrow(Data_ml))
Data_ml$Lag1008 <- head(c(rep(NA, 1008),
						head(Data_ml, nrow(Data_ml) - 1008)$Prices.BE), nrow(Data_ml))
Data_ml$Lag1176 <- head(c(rep(NA, 1176),
						head(Data_ml, nrow(Data_ml) - 1176)$Prices.BE), nrow(Data_ml))
Data_ml$Lag1344 <- head(c(rep(NA, 1344),
						head(Data_ml, nrow(Data_ml) - 1344)$Prices.BE), nrow(Data_ml))

profilesmin <- ddply(na.omit(Data_ml[,-1]), .(Month,DateType, Hour), colwise(min)) 
profilesmax <- ddply(na.omit(Data_ml[,-1]), .(Month,DateType, Hour), colwise(max))
profilesmean <- ddply(na.omit(Data_ml[,-1]), .(Month,DateType, Hour), colwise(mean))
Data_ml$PricesMin <- NA
Data_ml$PricesMax <- NA
for (i in 1:59808) {
  Data_ml$PricesMin[i] <- profilesmin[(profilesmin$Hour == Data_ml$Hour[i]) &
  									  (profilesmin$DateType == Data_ml$DateType[i]) &
  									  (profilesmin$Month == Data_ml$Month[i]),]$Prices.BE
  Data_ml$PricesMax[i] <- profilesmax[(profilesmax$Hour == Data_ml$Hour[i]) &
  									  (profilesmax$DateType == Data_ml$DateType[i]) &
  									  (profilesmax$Month == Data_ml$Month[i]),]$Prices.BE
  Data_ml$PricesCent[i] <- Data_ml$PricesMin[i] + (Data_ml$PricesMax[i] - Data_ml$PricesMin[i]) / 2
  Data_ml$PricesMean[i] <- profilesmean[(profilesmean$Hour == Data_ml$Hour[i]) &
  										(profilesmean$DateType == Data_ml$DateType[i]) &
  										(profilesmean$Month == Data_ml$Month[i]),]$Prices.BE
}  
Data_ml <- na.omit(Data_ml)  # Delete NAs

insample_ml <- head(Data_ml, nrow(Data_ml) - fh)  # insample for training
outsample_ml <- tail(Data_ml, fh)  # outsample for testing

Inspect Correlations
library(corrplot)
corrplot(cor(insample_ml[,-c(1,6)]), method="color")

# Year, Month, Hour, WeekDay, Holidays, Lag24, Lag168, Lag336, Lag504, Lag672, Lag840
# Lag1008, Lag1176
ml_model <- lm(Prices.BE ~ Year + Month + Hour + Weekday +
			   holidaysBE + Lag24 + Lag168 + Lag336 + Lag504 +
			   Lag672 + Lag840 + Lag1008 + Lag1176, data=insample_ml)
frc6_30 <- predict(ml_model, outsample_ml)
mean(200 * abs(outsample_ml$Prices.BE - frc6_30) / (abs(outsample_ml$Prices.BE) + abs(frc6_30)))

# Define final MLR
frc6 <- frc6_30
Evaluation$sMAPE[6] <- mean(200 * abs(outsample - frc6) / (abs(outsample) + abs(frc6)))

# Inspect MLR
plot(outsample, type="l", main="MLR") 
lines(frc6 + outsample-outsample, col="red", type="l")

# NN
normalize <- function(x) { return ((x - min(x)) / (max(x) - min(x))) }
ForScaling <- rbind(insample_ml, outsample_ml)[,c("Year", "Month", "Hour",
                                               "Weekday", "holidaysBE",
                                               "Lag168", "Lag336")]
ForScaling <- as.data.frame(lapply(ForScaling, normalize))

trainNN <- head(ForScaling, nrow(ForScaling) - fh)
validateNN <- normalize(insample_ml$Prices.BE)
testNN <- tail(ForScaling, fh)

size = 40
maxit = 100
initFunc = "RBF_Weights"
learnFunc = "BackpropWeightDecay"
hiddenActFunc = "Act_Logistic"
shufflePatterns = FALSE

model12 <- mlp(trainNN, validateNN,
               size=size, maxit=maxit, initFunc=initFunc,
               learnFunc=learnFunc, hiddenActFunc=hiddenActFunc,
               shufflePatterns=shufflePatterns, linOut=FALSE)

frc7_12 <- as.numeric(predict(model12, testNN)) *
					  (max(insample_ml$Prices.BE) - min(insample_ml$Prices.BE)) +
					   min(insample_ml$Prices.BE)
sMAPE_m12 <- mean(200 * abs(outsample - frc7_12) / (abs(outsample) + abs(frc7_12)))

frc7 <- frc7_12

# Inspect 
plot(outsample, type="l", main="NN")
lines(frc7 + outsample - outsample, col="red", type="l")

Evaluation$sMAPE[7] <- mean(200 * abs(outsample-frc7) / (abs(outsample) + abs(frc7)))

# Holt
# SMAPE
calculate.smape <- function(frc, outsample) {
  mean(200 * abs(outsample - frc) / (abs(outsample) + abs(frc)))  
}

Indexes_in <- decompose(insample, type="multiplicative")$seasonal
Indexes_out <- as.numeric(tail(Indexes_in, fh))
insample_no_seasonal <- insample / Indexes_in  # insample

EvaluationHolt <- data.frame(matrix(NA, ncol=3, nrow=100))
colnames(EvaluationHolt) <- c("a", "b", "sMAPE")

x <- c(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09)
y <- c(0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009)

i <- 1
for (valx in x) {
 for (valy in y) {
   if (valy < valx) {
     EvaluationHolt$a[i] <- valx
     EvaluationHolt$b[i] <- valy
     hw <- HoltWinters(insample_no_seasonal, gamma=FALSE, alpha=valx, beta=valy)
     frc <- forecast(hw, h=fh)$mean * Indexes_out
     EvaluationHolt$sMAPE[i] <- calculate.smape(frc, outsample)
     i <- i + 1
   }
 }
}

hw <- HoltWinters(insample_no_seasonal, gamma=FALSE, alpha=0.05, beta=0.009)
frc8 <- forecast(hw, h=fh)$mean * Indexes_out

plot(insample, type="l", ylab="Price", col=2)
lines(frc8, type="l", ylab="Price", col=3)
Evaluation$sMAPE[8] <- calculate.smape(frc8, outsample)
write_xlsx(EvaluationHolt, path="sMAPE_Holt.xlsx", col_names=TRUE)


# Random forest 
# Only Year, Month, Hour, Day, Holidays and Generation
Data_rf <- train
Test_rf <- test
Data_rf$Year <- year(Data_rf$datetime_utc)  # Define Year
Data_rf$Month <- month(Data_rf$datetime_utc)  # Define Month
Data_rf$Weekday <- 1
Data_rf[(Data_rf$DateType == 1) | (Data_rf$DateType == 7),]$Weekday <- 0
Data_rf$Lag168 = Data_rf$Lag336 <- NA  # Define Level
Data_rf$Lag168 <- head(c(rep(NA, 168),
					   head(Data_rf, nrow(Data_rf) - 168)$Prices.BE), nrow(Data_rf))
Data_rf$Lag336 <- head(c(rep(NA, 336), head(Data_rf, nrow(Data_rf) -336)$Prices.BE), nrow(Data_rf))
Data_rf <- na.omit(Data_rf)  # Delete NAs
Test_rf$Year <- year(Test_rf$datetime_utc)  # Define Year
Test_rf$Month <- month(Test_rf$datetime_utc)  # Define Month
Test_rf$Weekday <- 1
Test_rf[(Test_rf$DateType == 1) | (Test_rf$DateType == 7),]$Weekday <- 0
Test_rf$Lag168 = Test_rf$Lag336 <- NA  # Define Level
Test_rf$Lag168 <- head(c(rep(NA, 168),
					   head(Data_rf, nrow(Data_rf) - 168)$Prices.BE), nrow(Data_rf))
Test_rf$Lag336 <- head(c(rep(NA, 336),
					   head(Data_rf, nrow(Data_rf) - 336)$Prices.BE), nrow(Data_rf))
Data_rf <- na.omit(Data_rf)  # Delete NAs
insample_rf <- head(Data_rf, nrow(Data_rf))  # insample for training
outsample_rf <- tail(Test_rf, fh)  # outsample for testing

rf_model31 <- randomForest(Prices.BE ~ Year + Month + Hour +
                           Weekday + holidaysBE + 
                           Generation_FR + Generation_BE,
                           data=insample_rf, ntree=500, mtry=3, importance=TRUE)
frc9 <- predict(rf_model31, outsample_rf)
Evaluation$sMAPE[9] <- mean(200 *
							abs(outsample_rf$Prices.BE - frc9) /
							(abs(outsample_rf$Prices.BE) + abs(frc9)))


# Gradient Boosted Machine with all Lags
Data_gbm2 <- train
Data_gbm2$Year <- year(Data_gbm2$datetime_utc)  # Define Year
Data_gbm2$Month <- month(Data_gbm2$datetime_utc)  # Define Month
Data_gbm2$Weekday <- 1
Data_gbm2[(Data_gbm2$DateType == 1) | (Data_gbm2$DateType == 7),]$Weekday <- 0
Data_gbm2$Lag168 = Data_gbm$Lag336 <- NA  # Define Level
Data_gbm2$Lag24 = Data_gbm2$Lag168 = Data_gbm2$Lag336 = Data_gbm2$Lag504 <- NA
Data_gbm2$Lag672 = Data_gbm2$Lag840 = Data_gbm2$Lag1008 = Data_gbm2$Lag1176 <-NA
Data_gbm2$Lag1344 <- NA  # Define Level
Data_gbm2$Lag24 <- head(c(rep(NA, 24),
						head(Data_gbm2, nrow(Data_gbm2) - 24)$Prices.BE), nrow(Data_gbm2))
Data_gbm2$Lag168 <- head(c(rep(NA, 168),
						 head(Data_gbm2, nrow(Data_gbm2) - 168)$Prices.BE), nrow(Data_gbm2))
Data_gbm2$Lag336 <- head(c(rep(NA, 336),
						 head(Data_gbm2, nrow(Data_gbm2) - 336)$Prices.BE), nrow(Data_gbm2))
Data_gbm2$Lag504 <- head(c(rep(NA, 504),
						 head(Data_gbm2, nrow(Data_gbm2) - 504)$Prices.BE), nrow(Data_gbm2))
Data_gbm2$Lag672 <- head(c(rep(NA, 672),
						 head(Data_gbm2, nrow(Data_gbm2) - 672)$Prices.BE), nrow(Data_gbm2))
Data_gbm2$Lag840 <- head(c(rep(NA, 840),
						 head(Data_gbm2, nrow(Data_gbm2) - 840)$Prices.BE), nrow(Data_gbm2))
Data_gbm2$Lag1008 <- head(c(rep(NA, 1008),
						  head(Data_gbm2, nrow(Data_gbm2) - 1008)$Prices.BE), nrow(Data_gbm2))
Data_gbm2$Lag1176 <- head(c(rep(NA, 1176),
						  head(Data_gbm2, nrow(Data_gbm2) - 1176)$Prices.BE), nrow(Data_gbm2))
Data_gbm2$Lag1344 <- head(c(rep(NA, 1344),
						  head(Data_gbm2, nrow(Data_gbm2) - 1344)$Prices.BE), nrow(Data_gbm2))
Data_gbm2$Lag168 <- head(c(rep(NA, 168),
						 head(Data_gbm, nrow(Data_gbm) - 168)$Prices.BE), nrow(Data_gbm))
Data_gbm$Lag336 <- head(c(rep(NA, 336),
						head(Data_gbm, nrow(Data_gbm) - 336)$Prices.BE), nrow(Data_gbm))
Data_gbm2 <- na.omit(Data_gbm2)  # Delete NAs
insample_gbm2 <- head(Data_gbm2, nrow(Data_gbm2) - fh)  # insample for training
outsample_gbm2 <- tail(Data_gbm2, fh)  # outsamplefor testing

# Only Year, Month, Hour, Day, Holidays, all Lags
gbm_model6 <- gbm(Prices.BE ~ Year + Month + Hour +
                  Weekday + holidaysBE +
                  Lag24 + Lag168 + Lag336 + Lag504 +
                  Lag672 + Lag840 + Lag1008 + Lag1176, data=insample_gbm2, distribution="gaussian")
frc10 <- predict(gbm_model6, outsample_gbm2, n.trees=gbm_model6$n.trees)
Evaluation$sMAPE[10] <- mean(200 *
							 abs(outsample_gbm2$Prices.BE - frc10) /
							 (abs(outsample_gbm2$Prices.BE) + abs(frc10))) 


# Comb
frc11 <- (as.numeric(frc8) + frc9) / 2

test$Prices.BE <- frc11

# Decide which method to use and apply changes
j <- 0
for (i in 59497:59665) {
  j <- j + 1
  if (is.na(train_all$Prices.BE[i]) == T) {
    train_all$Prices.BE[i] <- frc11[j]
  }
}
missing <- train_all[is.na(train_all$Prices.BE) == T,]
nrow(missing)


write.table (train_all[,1:5],
			 file="TrainSet2.csv", append=FALSE, quote=TRUE,
			 sep=",", row.names=FALSE, col.names=TRUE, dec=".")

Evaluation$sMAPE[11] <- mean(200 * abs(outsample - frc11) / (abs(outsample) + abs(frc11)))
plot(insample, type="l", main="Comb")
lines(frc6 + outsample - outsample, col=2, type="l")  # MLR
lines(frc7 + outsample - outsample, col=3, type="l")  # NN
lines(frc8 + outsample - outsample, col=4, type="l")  # HOLT
lines(frc9 + outsample - outsample, col=5, type="l")  # Random Forest
lines(frc10 + outsample - outsample, col=6, type="l")  # GBM
lines(frc11 + outsample - outsample, col=6, type="l")  # COMB
legend("topleft", legend=c("HOLT", "Random Forest", "Comb"), col=c(4:6), lty=1, cex=0.8)

# Explain the effect of comb
ME1 <- outsample - frc6
ME2 <- outsample - frc7
ME3 <- outsample - frc8 
ME4 <- outsample - frc9
ME5 <- outsample - frc10
ME6 <- outsample - frc11
bias <- c(mean(ME1), mean(ME2), mean(ME3), mean(ME4), mean(ME5), mean(ME6))
acc <- c(mean(abs(ME1)),
		 mean(abs(ME2)), mean(abs(ME3)), mean(abs(ME4)), mean(abs(ME5)), mean(abs(ME6)))
plot(bias, acc, ylim=c(14, 18), xlim=c(0, 15))
text(bias, acc, c("MLR", "NN", "HOLT", "Random Forest", "GBM", "Comb"), col="red", pos=1)
