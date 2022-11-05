# Yuvaraj, 303384

import sys
dict = {}
#Partitoner - http://rare-chiller-615.appspot.com/mr1.html
for line in sys.stdin:
    line = line.strip()
    line = line.split('\t')  # Tab separated
    dep_airport = line[0]
    dep_delay = line[1]
    if dep_airport in dict:  # Key-value pairs
        dict[dep_airport].append(float(dep_delay))
    else:
        dict[dep_airport] = []
        dict[dep_airport].append(float(dep_delay))
#Reducer
for dep_airport in dict.keys():
    avg_delay = sum(dict[dep_airport])*1.0 / len(dict[dep_airport])
    max_delay = max(dict[dep_airport])
    min_delay = min(dict[dep_airport])
    string = '%s\t%s\t%s\t%s' % (dep_airport, avg_delay,max_delay,min_delay)
    print(string)
