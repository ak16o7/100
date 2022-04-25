import pip
pip.main(["install","matplotlib"])
import matplotlib.pyplot as plt

a = [1,2,3,4,5,6,7,8,9,10]
b = [31,42,15,-2,4,512,21,25,19,95]

plt.scatter(a,b)
plt.show()