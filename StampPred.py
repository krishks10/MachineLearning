import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

stamp_bought = np.array([1,3,5,7,9]).reshape(-1,1)
amount_paid = np.array([2,6,8,12,18])

model = LinearRegression()

model.fit(stamp_bought, amount_paid)

next_month_stamps = 10
predicted_amount = model.predict([[next_month_stamps]])

print("Predicted amount for 10 stamps:", predicted_amount[0])  

plt.scatter(stamp_bought, amount_paid, color = 'blue')
plt.plot(stamp_bought, model.predict(stamp_bought), color = 'red')
plt.xlabel('Stamps Bought')
plt.ylabel('Amount Paid')
plt.title('Stamp Purchase Prediction')
plt.show()