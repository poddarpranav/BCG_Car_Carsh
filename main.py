from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, row_number

from config import input_path
from utils import load_data

import os
import sys

class Accidents:
    def __init__(self, config_file,spark):
        input_file_path=input_path
        self.df_charges=load_data(spark,input_file_path,"Charges")
        self.df_damages = load_data(spark,input_file_path,"Damages")
        self.df_endorse = load_data(spark,input_file_path,"Endorse")
        self.df_primary_person = load_data(spark,input_file_path,"Primary_Person")
        self.df_units = load_data(spark,input_file_path,"Units")
        self.df_restrict = load_data(spark,input_file_path,"Restrict")

    def male_accidents(self):
        """
        Find the number of crashes (accidents) in which number of persons killed are mal

        """
        df = self.df_primary_person.filter(self.df_primary_person.PRSN_GNDR_ID == "MALE")
        return df.count()

    def two_wheeler_accidents(self):
        """
        Find no. of two wheelers booked for crashes.
        """
        df = self.df_units.filter(col("VEH_BODY_STYL_ID").contains("MOTORCYCLE"))
        return df.count()

    def highest_female_accident_state(self):
        """
        Finds state with highest female accidents

        """
        df = self.df_primary_person.filter(self.df_primary_person.PRSN_GNDR_ID == "FEMALE"). \
            groupby("DRVR_LIC_STATE_ID").count(). \
            orderBy(col("count").desc())

        return df.first().DRVR_LIC_STATE_ID

    def top_vehicle_contributing_to_injuries(self):
        """
        Find Top 5th to 15th VEH_MAKE_IDs that contribute to a largest number of injuries including death
        """
        df = self.df_units.filter(self.df_units.VEH_MAKE_ID != "NA"). \
            withColumn('TOT_CASUALTIES_CNT', self.df_units['TOT_INJRY_CNT'] + self.df_units['DEATH_CNT']). \
            groupby("VEH_MAKE_ID").sum("TOT_CASUALTIES_CNT"). \
            withColumnRenamed("sum(TOT_CASUALTIES_CNT)", "TOT_CASUALTIES_CNT_AGG"). \
            orderBy(col("TOT_CASUALTIES_CNT_AGG").desc())

        df_top_5_15 = df.limit(15).subtract(df.limit(5))
        return [res[0] for res in df_top_5_15.select("VEH_MAKE_ID").collect()]


    def top_ethnic_group_per_body_style(self):
        """
        Find top ethnic user group of each unique body style that was involved in crashes
        """
        w = Window.partitionBy("VEH_BODY_STYL_ID").orderBy(col("count").desc())
        df = self.df_units.join(self.df_primary_person, on=['CRASH_ID'], how='inner'). \
            filter(~self.df_units.VEH_BODY_STYL_ID.isin(["NA", "UNKNOWN", "NOT REPORTED",
                                                         "OTHER  (EXPLAIN IN NARRATIVE)"])). \
            filter(~self.df_primary_person.PRSN_ETHNICITY_ID.isin(["NA", "UNKNOWN"])). \
            groupby("VEH_BODY_STYL_ID", "PRSN_ETHNICITY_ID").count(). \
            withColumn("row", row_number().over(w)).filter(col("row") == 1).drop("row", "count")
        

        df.show(truncate=False)
      
        return df

    def top_5_zip_codes_with_alcohols_as_factor(self):
        """
        Find top 5 Zip Codes with the highest number crashes with alcohols as the contributing factor to a crash

        """
        df = self.df_units.join(self.df_primary_person, on=['CRASH_ID'], how='inner'). \
            dropna(subset=["DRVR_ZIP"]). \
            filter(col("CONTRIB_FACTR_1_ID").contains("ALCOHOL") | col("CONTRIB_FACTR_2_ID").contains("ALCOHOL")). \
            groupby("DRVR_ZIP").count().orderBy(col("count").desc()).limit(5)

        return [res[0] for res in df.collect()]

    def crash_ids_with_no_damage(self):
        """
        Count Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~) is above 4
        and car avails Insurance.

        """
        df = self.df_damages.join(self.df_units, on=["CRASH_ID"], how='inner'). \
            filter(
            (
                    (self.df_units.VEH_DMAG_SCL_1_ID > "DAMAGED 4") &
                    (~self.df_units.VEH_DMAG_SCL_1_ID.isin(["NA", "NO DAMAGE", "INVALID VALUE"]))
            ) | (
                    (self.df_units.VEH_DMAG_SCL_2_ID > "DAMAGED 4") &
                    (~self.df_units.VEH_DMAG_SCL_2_ID.isin(["NA", "NO DAMAGE", "INVALID VALUE"]))
            )
        ). \
            filter(self.df_damages.DAMAGED_PROPERTY == "NONE"). \
            filter(self.df_units.FIN_RESP_TYPE_ID == "PROOF OF LIABILITY INSURANCE")

        return [row[0] for row in df.collect()]

    def top_5_vehicle_brand(self):
        """
        Top 5 Vehicle Makes/Brands where drivers are charged with speeding related offences, has licensed
        Drivers, uses top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of
        offences

        """
        top_25_state_list = [row[0] for row in self.df_units.filter(col("VEH_LIC_STATE_ID").cast("int").isNull()).
        groupby("VEH_LIC_STATE_ID").count().orderBy(col("count").desc()).limit(25).collect()]
        top_10_used_vehicle_colors = [row[0] for row in self.df_units.filter(self.df_units.VEH_COLOR_ID != "NA").
        groupby("VEH_COLOR_ID").count().orderBy(col("count").desc()).limit(10).collect()]

        df = self.df_charges.join(self.df_primary_person, on=['CRASH_ID'], how='inner'). \
            join(self.df_units, on=['CRASH_ID'], how='inner'). \
            filter(self.df_charges.CHARGE.contains("SPEED")). \
            filter(self.df_primary_person.DRVR_LIC_TYPE_ID.isin(["DRIVER LICENSE", "COMMERCIAL DRIVER LIC."])). \
            filter(self.df_units.VEH_COLOR_ID.isin(top_10_used_vehicle_colors)). \
            filter(self.df_units.VEH_LIC_STATE_ID.isin(top_25_state_list)). \
            groupby("VEH_MAKE_ID").count(). \
            orderBy(col("count").desc()).limit(5)

        return [row[0] for row in df.collect()]


if __name__ == '__main__':
    # Initialize sparks session
    spark = SparkSession \
        .builder \
        .appName("Car_Crash Analysis") \
        .getOrCreate()

    config_file = "config.py"
    spark.sparkContext.setLogLevel("ERROR")


    accident = Accidents(config_file,spark)
    
    """
        1.Find the number of crashes (accidents) in which number of persons killed are male

    """
    res1=accident.male_accidents()
    print("1.Number of crashes (accidents) in which number of persons killed are male:\n", res1)
    

    """
        2.Find no. of two wheelers booked for crashes.
    """
    res2=accident.two_wheeler_accidents()
    print("\n\n 2.No. of two wheelers booked for crashes:\n",res2 )
 

    """
       3.Finds state with highest female accidents

    """
    res3=accident.highest_female_accident_state()
    print("\n\n 3. State with highest female accidents:\n",res3)
   

    """
        4.Find Top 5th to 15th VEH_MAKE_IDs that contribute to a largest number of injuries including death
    """
    res4=accident.top_vehicle_contributing_to_injuries()
    print("\n\n4. Top 5th to 15th VEH_MAKE_IDs that contribute to a largest number of injuries including death:\n", ','.join(res4))
    

    """
       5. Find top ethnic user group of each unique body style that was involved in crashes
    """
    print("\n\n5.Top ethnic user group of each unique body style that was involved in crashes:\n")
    res5=accident.top_ethnic_group_per_body_style()
    

    """
        6.Find top 5 Zip Codes with the highest number crashes with alcohols as the contributing factor to a crash

     """
    res6=accident.top_5_zip_codes_with_alcohols_as_factor()
    print("\n\n6.Top 5 Zip Codes with the highest number crashes with alcohols as the contributing factor to a crash:\n", ','.join(res6))
  

    """
        7.Count Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~) is above 4
        and car avails Insurance.

    """
    res7=accident.crash_ids_with_no_damage()
    print("\n\n7.Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~) is above 4 and car avails Insurance:\n", ','.join(map(str, res7)))
    

    """
        8.Top 5 Vehicle Makes/Brands where drivers are charged with speeding related offences, has licensed
        Drivers, uses top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of
        offences

    """
    res8=accident.top_5_vehicle_brand()
    print('''\n\n8. Top 5 Vehicle Makes/Brands where drivers are charged with speeding related offences, has licensed Drivers, uses top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of offences\n''', ','.join(res8))
   

spark.stop()