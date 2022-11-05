# Yuvaraj, 303384

import sys
dict = {}
#Partitoner - http://rare-chiller-615.appspot.com/mr1.html
for line in sys.stdin:
    line = line.strip()
    line = line.split('\t')  # Tab separated
    arr_airport = line[0]
    arr_delay = line[1]
	if arr_delay != "\n":
		if arr_airport in dict:  # Key-value pairs
			dict[arr_airport].append(float(arr_delay))
		else:
			dict[arr_airport] = []
			dict[arr_airport].append(float(arr_delay))
#Reducer
avg_delay = {}        
for arr_airport in dict.keys():
    average_delay = sum(dict[arr_airport])*1.0 / len(dict[arr_airport])
    avg_delay[arr_airport] = average_delay
       
arr_airport = avg_delay.keys()
dict = avg_delay.values()    
topten = sorted(range(len(dict)), key=lambda i: dict[i])[-10:]	
	
for i in topten: 
    print(arr_airport[i],dict[i]) 
