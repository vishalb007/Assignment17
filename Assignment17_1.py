import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def advertising():
	advertising_data=pd.read_csv("Advertising.csv",index_col=0)
	print(advertising_data.head())

	X=pd.DataFrame(advertising_data,columns=['TV','radio','newspaper'])
	Y=pd.DataFrame(advertising_data,columns=['sales'])

	xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.5)

	linmodel=LinearRegression()
	linmodel=linmodel.fit(xtrain,ytrain)

	y_pred=linmodel.predict(xtest)

	print("Expected ",ytest)

	df = pd.DataFrame(y_pred, columns = ['Obtained Ouput'])
	print(df)

def main():
	advertising()

if __name__=="__main__":
	main()	