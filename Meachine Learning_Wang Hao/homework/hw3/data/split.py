import numpy as np
# Read csv file
data = np.loadtxt("kc_house_data.csv", delimiter=",", dtype=str, skiprows=1) #, max_rows=1000
# Select columns by index
selected_columns_X = data[:, [2, 3, 4, 5, 6, 10, 11, 12, 14, 19, 20]]
# Save new csv file

np.savetxt("train.csv", selected_columns_X, delimiter=",", fmt="%s")# Select columns by index

# The row of labels
'''
labels = "id,date,price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,zipcode,lat,long,sqft_living15,sqft_lot15"
# Split the labels by comma
labels = labels.split(",")
# Loop through the labels and numbers
for i, label in enumerate(labels):
  # Print the label and number
  print(label + ":" + str(i))

id:0
date:1
price:2
bedrooms:3
bathrooms:4
sqft_living:5
sqft_lot:6
floors:7
waterfront:8
view:9
condition:10
grade:11
sqft_above:12
sqft_basement:13
yr_built:14
yr_renovated:15
zipcode:16
lat:17
long:18
sqft_living15:19
sqft_lot15:20

'''