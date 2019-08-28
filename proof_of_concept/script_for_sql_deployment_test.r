library(data.table)
library(stringr)
library(PerformanceAnalytics)
library(caret)
library(mlbench)
library(kernlab)
library(xgboost)
library(foreach)
library(dplyr)

setwd("~/R Projects/Auction_Price_Predictor/proof_of_concept/") # set working directory, probably not necessary if you already have the connection established

# read prediction algorithms .... will need to change the path
fld_gbm <- readRDS('dynaglide_model_gbm_shortV_20180322.rds')
fld_rf <- readRDS('dynaglide_model_rf_shortV_20180321.rds')
flx_gbm <- readRDS('softail_model_gbm_shortV_20180321.rds')
flx_rf <- readRDS('softail_model_rf_shortV_20180321.rds')
tour_gbm <- readRDS('touring_model_gbm_shortV_20180321.rds')
tour_rf <- readRDS('touring_model_rf_shortV_20180321.rds')
xl_gbm <- readRDS('sportster_model_gbm_shortV_20180321.rds')
xl_rf <- readRDS('sportster_model_rf_shortV_20180321.rds')

# read lookup tables
mdllkp <- fread('model_lookup.csv', header = T, strip.white = T, stringsAsFactors = T)
dsplkp <- fread('displacement_lookup.csv', header = T, strip.white = T, stringsAsFactors = T)
plntlkp <- fread('plant_lookup.csv', header = T, strip.white = T, stringsAsFactors = T)
color <- read.csv('color_category.csv', header = T, strip.white = T, stringsAsFactors = F)


# query DB for recent

# read data
hd <- fread("~/R Projects/Auction_Price_Predictor/proof_of_concept/HarleyData_20171005/HarleyData_20171220.txt", sep = '|', fill = T, verbose = T, header = T, strip.white = T, blank.lines.skip = T, stringsAsFactors = F)
hd <- hd[1:193051,]
hd <- hd[,c(1:99, 102:104)]

hd2 <- hd[which(hd$LiveSoldSimulcast == 1 | hd$LiveSoldSimulcast == 0 | hd$LiveSoldSimulcast == ''),]
hd2 <- hd2[which(hd2$Value != ''),]
hd2 <- hd2[which(hd2$LiveSoldSimulcast == 1 | hd2$LiveSoldSimulcast == 0),]

hd3 <- hd2[which(hd2$VehicleCategory == 'CRUISER'),]

hd3[, c(16, 26:28, 33:43, 46, 48, 51, 62:67, 93, 95, 97, 99:101)] <- lapply(hd3[, c(16, 26:28, 33:43, 46, 48, 51, 62:67, 93, 95, 97, 99:101)], as.character)
hd3[, c(16, 26:28, 33:43, 46, 48, 51, 62:67, 93, 95, 97, 99:101)] <- lapply(hd3[, c(16, 26:28, 33:43, 46, 48, 51, 62:67, 93, 95, 97, 99:101)], as.numeric)

hd3 <- hd3[which(hd3$Year >= 1981),]

###################################################### VIN decoding
hd3$vinMarket <- as.factor(toupper(str_sub(hd3$VIN, 1, 3)))
hd3$vinWeight <- as.factor(str_sub(hd3$VIN, 4, 4))
hd3$vinModel <- as.factor(str_sub(hd3$VIN, 5, 6))
hd3$vinDisplacement <- as.factor(str_sub(hd3$VIN, 7, 7))
hd3$vinIntroduction <- as.factor(str_sub(hd3$VIN, 8, 8))
hd3$vinPlant <- as.factor(str_sub(hd3$VIN, 11, 11))

##################################################### VIN labels
hd4 <- join(hd3, mdllkp, by = 'vinModel', type = 'left')
for (col in c("ModelDesignation2_3")) hd4[is.na(get(col)), (col) := 'NoMatch']

hd4 <- hd4[which(hd4$ModelDesignation2_3 != 'NoMatch')]
######### Vin name
hd4$ModelDesignation2p <- as.factor(paste(hd4$ModelDesignation2_3, hd4$ModelDesignation4p, sep = ' '))

####### min, mean, max Value amount
vmin <- aggregate(Value ~ ModelDesignation2p, hd4[which(hd4$Value > 0),], min)
vmean <- aggregate(Value ~ ModelDesignation2p, hd4[which(hd4$Value > 0),], mean)
vmedian <- aggregate(Value ~ ModelDesignation2p, hd4[which(hd4$Value > 0),], median)
vmax <- aggregate(Value ~ ModelDesignation2p, hd4[which(hd4$Value > 0),], max)
vsd <- aggregate(Value ~ ModelDesignation2p, hd4[which(hd4$Value > 0),], sd)

vrange <- join(vmin, vmean, by = 'ModelDesignation2p', type = 'left')
vrange <- join(vrange, vmedian, by = 'ModelDesignation2p', type = 'left')
vrange <- join(vrange, vmax, by = 'ModelDesignation2p', type = 'left')
vrange <- join(vrange, vsd, by = 'ModelDesignation2p', type = 'left')
names(vrange) <- c('ModelDesignation2p', 'vmin', 'vmean', 'vmedian', 'vmax', 'vsd')
hd4 <- join(hd4, vrange, by = 'ModelDesignation2p', type = 'left')
hd4$vz <- (hd4$Value - hd4$vmean) / hd4$vsd

############################## look at 2015+ auctions
hd5 <- hd4[which(hd4$ValueYear >= 2015),]
hd5 <- droplevels(hd5)
hd5 <- join(hd5, dsplkp, by = 'vinDisplacement', type = 'left')

# clean data for missing etc.
df <- hd5 # creating a new object for next stage, doesn't have to be done
df <- df[which(df$OriginalMSRP != 'NA' & df$Mileage != 'NA' & df$Score != 'NA' & df$Frame_Score != 'NA'),]
df <- df[which(df$ValueMethod == 'Sale'),]
df <- df[which(df$Value > 0),]
df <- df[which(df$MileagePerYear != 'NA'),]
df <- df[which(df$ReserveAmount != 'NA'),]
dfa <- df[which(df$AuctionID != 'NULL'),]



dfa <- join(dfa, color, by = 'Color', type = 'left')
dfa$VIN6 <- str_trunc(as.character(dfa$VIN), 6, side = 'right', ellipsis = '')

# set variables to factor data type
dfa[, c('AuctionID',
        'VehicleID',
        'VIN',
        'BusinessSegment',
        'IsSalvageUnit',
        'ValueMonth',
        'ValueQtr',
        'ValueLocation',
        'SourceType',
        'ModelDesignation',
        'ModelDesignation2_3',
        'ModelDesignation4p',
        'ModelDesignation2p',
        'edition',
        'Displacement',
        'colorCat',
        'vinMarket',
        'vinWeight',
        'vinDisplacement',
        'vinIntroduction',
        'vinPlant',
        'VIN6'
)] <- lapply(dfa[, c('AuctionID',
                     'VehicleID',
                     'VIN',
                     'BusinessSegment',
                     'IsSalvageUnit',
                     'ValueMonth',
                     'ValueQtr',
                     'ValueLocation',
                     'SourceType',
                     'ModelDesignation',
                     'ModelDesignation2_3',
                     'ModelDesignation4p',
                     'ModelDesignation2p',
                     'edition',
                     'Displacement',
                     'colorCat',
                     'vinMarket',
                     'vinWeight',
                     'vinDisplacement',
                     'vinIntroduction',
                     'vinPlant',
                     'VIN6'
)], as.factor)


# can't recall if you have
dfa$gbmPred <- ifelse(MacroModel() == 'FX/FL Dyna Glide', round(predict(fld_gbm, userdata), 0),
                  ifelse(MacroModel()== 'FL/FX Softail', round(predict(flx_gbm, userdata), 0),
                         ifelse(MacroModel()== 'FL Touring', round(predict(tour_gbm, userdata), 0),
                                ifelse(MacroModel()== 'XL Sportster', round(predict(xl_gbm, userdata), 0), 9999999
                                )
                         )
                  )
)