
library(jsonlite)
#Input csv from system
nlpdata <- read.csv(file="C://Users/csame/Desktop/AIT 624 Project/Data1.csv",header = FALSE)
nlpdata1 <- read.csv(file="C://Users/csame/Desktop/AIT 624 Project/location.csv",header = FALSE)

library(tidyr)
library(stringr)
library(foreign)

#Column Formatting - Removing unnecessary data
nlpdata1 <- separate(nlpdata1,V1,into = c("Unknown","location"),sep =".com/")
nlpdata1$location <- gsub("/home-values/","",nlpdata1$location)

#Defining new column
for(i in seq(from=1, to=108, by=1)){
  nlpdata$location <- NA
}

for(i in seq(from=1, to=108, by=1)){
  nlpdata$condodetail <- NA
}

#print(nlpdata)

#Cleaning the dataset
nlpdata$V1 <- gsub("@", " " , nlpdata$V1)
nlpdata$V1 <- gsub("#", " " , nlpdata$V1)
nlpdata$V1 <- gsub("(s?)(f|ht)tp(s?)://\\S+\\b", "", nlpdata$V1)
nlpdata$V1 <- gsub("[[:punct:]]", " " , nlpdata$V1)
nlpdata$V1 <- gsub("[0-9]", " " , nlpdata$V1)
nlpdata$V1 <- gsub("\\ can ", " " , nlpdata$V1)
nlpdata$V1 <- gsub("\\ to ", " " , nlpdata$V1)
nlpdata$V1 <- gsub("\\ is ", " " , nlpdata$V1)
nlpdata$V1 <- gsub("\\ the ", " " , nlpdata$V1)
nlpdata$V1 <- gsub("\\ did ", " " , nlpdata$V1)
nlpdata$V1 <- gsub("\\You ", " " ,nlpdata$V1)
nlpdata$V1 <- gsub("\\Your ", " " , nlpdata$V1)
nlpdata$V1 <- gsub("\\The ", " " , nlpdata$V1)
nlpdata$V1 <- tolower(nlpdata$V1)

#print(nlpdata)

#Copying the Location column
for (i in seq(from=1, to=108, by=1)){
  nlpdata$location <- nlpdata1$location
  
}

#Renaming the column
for (i in seq(from=1, to=108, by=1)){
  nlpdata$condodetail <- nlpdata$V1
}

#Dropping the column V1
for (i in seq(from=1, to=108, by=1)){
  nlpdata$V1 <- NULL
}

#Removing '-' from the location column
nlpdata$location <- gsub("-"," ",nlpdata$location)

#Saving the file as CSV
write.csv(nlpdata,"C:/Users/csame/Desktop/AIT 624 Project/624project.csv")
#exceldata = read.csv('C:/Users/csame/Desktop/624project.csv')
