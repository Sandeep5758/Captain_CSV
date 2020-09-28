# import dependencies
  import numpy as np
  import pandas as pd
  from matplotlib import pyplot as plt
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.cross_validation import train_test_split
  #matplotlib inline
  
  # using pandas to read the database stored in the same folder
     data = pd.read_CSV('mnist.csv')
     
  # viewing coloumn heads()
  data.head()
  
  # extracting data from the dataset and viewing them up close
    a = data.iloc[3,1:].values
    
  # reshaping the extracted data into reasonable size
    a = a.reshape(28,28).astype('unit8')
    plt.inshow(a)
    
  # Preparing the data
  # Separating labels and data values
  df_x = data.iloc[:,1:]
  df_y = data.iloc[:,0]
  
  # creating test and train Sizes/batches
  x_train, x_test, y_train, y-test = train_test_split(df_x,df_y,test_size = 0.2,random_state=4)
  
  #check data
  y_train.head()
  
  # call rf classifier
  rf = RandomForestClassifier(n_estimators=100)
  
  # fit the model
  rf.fit(x_train, y_train)
  
  # prediction on test data
  pred = rf.predict(x_test)
  
  pred
  
  # check prediction accuracy
  a = y_test.values
  
  # calculation number of correctly predicted values
  count = 0
  for i in range(len(pred)):
  if pred[i] == a[i]:
      count = count+1
      
  count
  
  # total values that prediction code was run on
  len(pred)
  # accuracy value
  
