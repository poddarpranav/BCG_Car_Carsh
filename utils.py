def load_data(spark, input_file_path,filename):
    df = spark.read.option("inferSchema", "true").csv(input_file_path[filename], header=True)
    return df