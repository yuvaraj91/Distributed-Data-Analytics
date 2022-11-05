## Yuvaraj, 303384
#http://rare-chiller-615.appspot.com/mr1.html 
import sys
for line in sys.stdin:
    line = line.strip().replace('\"', '')  # Remove leading, trailing whitespace. Remove quotes
    line = line.split(",")  # CSV
    if len(line) >= 2:
        arr_airport = line[4]  # Only getting the required columns
        arr_delay = line[8]
		if arr_delay == '':
			arr
        string = '%s\t%s' % (arr_airport, arr_delay)
        print(string)
