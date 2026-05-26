import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

pages = np.array([100, 150, 200, 250, 300, 350, 400,450, 500]).reshape(-1, 1)
likes = np.array([0,1,1,1,0,0,0,0,0])

model = LogisticRegression()

model.fit(pages, likes)

predict_page = 260
predicted_like = model.predict([[predict_page]])

plt.scatter(pages, likes, color='blue')
plt.plot(pages, model.predict_proba(pages)[:, 1], color='red')
plt.title("Pages vs Likes/Dislikes")
plt.xlabel("Number of Pages")
plt.ylabel("Like (1) / Dislike (0)")
plt.axvline(x=predict_page, color='green', linestyle='--')
plt.axhline(y=0.5, color='orange', linestyle='--')
plt.show()

print(f"Predicted like for {predict_page} pages: {'Like' if predicted_like[0] == 1 else 'Dislike'}")