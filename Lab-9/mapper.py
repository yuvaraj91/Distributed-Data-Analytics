## Yuvaraj, 303384
#http://rare-chiller-615.appspot.com/mr1.html 
import sys
for line in sys.stdin:
    line = line.strip().replace('\"', '')  # Remove leading, trailing whitespace. Remove quotes
    line = line.split(",")  # CSV
    if len(line) >= 2:
        dep_airport = line[3]  # Only getting the required columns
        dep_delay = line[6]
        string = '%s\t%s' % (dep_airport, dep_delay)
        print(string)
