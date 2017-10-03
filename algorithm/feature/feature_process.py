# -*- coding:utf-8 -*-

"""
@author: SJ
@software: PyCharm
@file: feature_process.py
@time: 2017-9-30 11:17
"""

class feature_process:

    def process(self, dataFrame):

        # life of property
        dataFrame["N-life"] = 2018 - dataFrame["yearbuilt"]
        # error in calculation of the finished living area of home
        dataFrame['N-LivingAreaError'] = dataFrame['calculatedfinishedsquarefeet'] / dataFrame['finishedsquarefeet12']

        # proportion of living area
        dataFrame['N-LivingAreaProp'] = dataFrame['calculatedfinishedsquarefeet'] / dataFrame['lotsizesquarefeet']
        dataFrame['N-LivingAreaProp2'] = dataFrame['finishedsquarefeet12'] / dataFrame['finishedsquarefeet15']

        # Amout of extra space
        dataFrame['N-ExtraSpace'] = dataFrame['lotsizesquarefeet'] - dataFrame['calculatedfinishedsquarefeet']
        dataFrame['N-ExtraSpace-2'] = dataFrame['finishedsquarefeet15'] - dataFrame['finishedsquarefeet12']

        # Total number of rooms
        dataFrame['N-TotalRooms'] = dataFrame['bathroomcnt'] * dataFrame['bedroomcnt']

        # Average room size
        dataFrame['N-AvRoomSize'] = dataFrame['calculatedfinishedsquarefeet'] / dataFrame['roomcnt']

        # Number of Extra rooms
        dataFrame['N-ExtraRooms'] = dataFrame['roomcnt'] - dataFrame['N-TotalRooms']

        # Ratio of the built structure value to land area
        dataFrame['N-ValueProp'] = dataFrame['structuretaxvaluedollarcnt'] / dataFrame['landtaxvaluedollarcnt']

        # Does property have a garage, pool or hot tub and AC?
        dataFrame['N-GarPoolAC'] = ((dataFrame['garagecarcnt'] > 0) & (dataFrame['pooltypeid10'] > 0) & (
        dataFrame['airconditioningtypeid'] != 5)) * 1

        dataFrame["N-location"] = dataFrame["latitude"] + dataFrame["longitude"]
        dataFrame["N-location-2"] = dataFrame["latitude"] * dataFrame["longitude"]
        dataFrame["N-location-2round"] = dataFrame["N-location-2"].round(-4)

        dataFrame["N-latitude-round"] = dataFrame["latitude"].round(-4)
        dataFrame["N-longitude-round"] = dataFrame["longitude"].round(-4)

        # Ratio of tax of property over parcel
        dataFrame['N-ValueRatio'] = dataFrame['taxvaluedollarcnt'] / dataFrame['taxamount']

        # TotalTaxScore
        dataFrame['N-TaxScore'] = dataFrame['taxvaluedollarcnt'] * dataFrame['taxamount']

        # polnomials of tax delinquency year
        # dataFrame["N-taxdelinquencyyear-2"] = dataFrame["taxdelinquencyyear"] ** 2
        # dataFrame["N-taxdelinquencyyear-3"] = dataFrame["taxdelinquencyyear"] ** 3

        # Length of time since unpaid taxes
        dataFrame['N-life'] = 2018 - dataFrame['taxdelinquencyyear']

        # Other features based off the location
        # Number of properties in the zip
        zip_count = dataFrame['regionidzip'].value_counts().to_dict()
        dataFrame['N-zip_count'] = dataFrame['regionidzip'].map(zip_count)

        # Number of properties in the city
        city_count = dataFrame['regionidcity'].value_counts().to_dict()
        dataFrame['N-city_count'] = dataFrame['regionidcity'].map(city_count)

        # Number of properties in the city
        region_count = dataFrame['regionidcounty'].value_counts().to_dict()
        dataFrame['N-county_count'] = dataFrame['regionidcounty'].map(city_count)

        # ------------------------------------------------------------
        # Let's create additional variables which are simplification of
        # some of the other variables

        # Indicator whether it has AC or not
        dataFrame['N-ACInd'] = (dataFrame['airconditioningtypeid'] != 5) * 1

        # Indicator whether it has Heating or not
        dataFrame['N-HeatInd'] = (dataFrame['heatingorsystemtypeid'] != 13) * 1

        # There's 25 different property uses - let's compress them down to 4 categories
        # dataFrame['N-PropType'] = dataFrame.propertylandusetypeid.replace(
        #     {31: "Mixed", 46: "Other", 47: "Mixed", 246: "Mixed", 247: "Mixed", 248: "Mixed", 260: "Home", 261: "Home",
        #      262: "Home", 263: "Home", 264: "Home", 265: "Home", 266: "Home", 267: "Home", 268: "Home",
        #      269: "Not Built", 270: "Home", 271: "Home", 273: "Home", 274: "Other", 275: "Home", 276: "Home",
        #      279: "Home", 290: "Not Built", 291: "Not Built"})

        # ------------------------------------------------------
        # One of the EDA kernels indicated that structuretaxvaluedollarcnt was
        # one of the most important features.So let's create some additional
        # variables on that.

        # polnomials of the variable
        dataFrame["N-structuretaxvaluedollarcnt-2"] = dataFrame["structuretaxvaluedollarcnt"] ** 2
        dataFrame["N-structuretaxvaluedollarcnt-3"] = dataFrame["structuretaxvaluedollarcnt"] ** 3

        # Average structuretaxvaluedollarcnt by city
        group = dataFrame.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
        dataFrame['N-Avg-structuretaxvaluedollarcnt'] = dataFrame['regionidcity'].map(group)

        # Deviation away from average
        dataFrame['N-Dev-structuretaxvaluedollarcnt'] = abs(
            (dataFrame['structuretaxvaluedollarcnt'] - dataFrame['N-Avg-structuretaxvaluedollarcnt'])) / dataFrame[
                                                           'N-Avg-structuretaxvaluedollarcnt']

        return dataFrame