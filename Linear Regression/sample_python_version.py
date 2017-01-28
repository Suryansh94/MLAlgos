## It is used to visualise the data

import graphlab
import matplotlib.pyplot as plt
sales=graphlab.SFrame('home_data.gl/')
graphlab.canvas.set_target('ipynb')
sales.show(view="Scatter Plot",x="sqft_living",y="price")
train_data,test_data=sales.random_split(0.8,0) # 8. percent training and 0 is seed
sqft_model=graphlab.linear_regression.create(train_data,target="price",features=["sqft_living"])
print sqft_model.evaluate(test_data)
sqft_model.get('coefficients')
plt.plot(test_data['sqft_living'],test_data['price'],'.',
         test_data['sqft_living'],sqft_model.predict(test_data),'-') # . used to print them as dot