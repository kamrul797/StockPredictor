import matplotlib.pyplot as plt;
import pandas as pd;

data=pd.read_csv(r'data/Actual.csv', parse_dates=True)
print(data.info());
BATBC= data["BATBC"];
BEXIMCO = data["BEXIMCO"];
LANKABANGLA = data["LANKABANGLA"]

fig = plt.figure(figsize=(10,5))
fig.suptitle('Actual Stock Closing Price [BATBC]', fontsize='12')
plt.plot(BATBC[400:],color='green',alpha=0.9, label="BATBC")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('Number of Days',fontsize='11.5')
plt.ylabel('Stock Closing Price',fontsize='11.5')
plt.show()
fig.savefig('Output/BATBC-Actual.jpg')

fig = plt.figure(figsize=(10,5))
fig.suptitle('Actual Stock Closing Price [LANKABANGLA]', fontsize='12')
plt.plot(LANKABANGLA[400:],color='green',alpha=0.9, label="LANKABANGLA")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('Number of Days',fontsize='11.5')
plt.ylabel('Stock Closing Price',fontsize='11.5')
plt.show()
fig.savefig('Output/LANKABANGLA-Actual.jpg')

fig = plt.figure(figsize=(10,5))
fig.suptitle('Actual Stock Closing Price [BEXIMCO]', fontsize='12')
plt.plot(BEXIMCO[400:],color='green',alpha=0.9, label="BEXIMCO")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('Number of Days',fontsize='11.5')
plt.ylabel('Stock Closing Price',fontsize='11.5')
plt.show()
fig.savefig('Output/BEXIMCO-Actual.jpg')